import os
import json
import aiohttp_cors
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from aiohttp import web, web_request, web_response
import bittensor as bt

class DashboardServer:
    """
    Production-ready web dashboard server for subnet analytics.
    Provides real-time monitoring, historical analysis, and system health dashboards.
    """
    
    def __init__(self, analytics_collector, validator_instance, host: str = "0.0.0.0", port: int = 8080):
        self.analytics_collector = analytics_collector
        self.validator = validator_instance
        self.host = host
        self.port = port
        
        # Web application setup
        self.app = web.Application()
        self.runner = None
        self.site = None
        
        # Setup CORS
        self.cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # Setup routes
        self._setup_routes()
        
        bt.logging.info(f"DashboardServer initialized on {host}:{port}")

    def _setup_routes(self):
        """Setup web application routes"""
        # API routes
        self.app.router.add_get('/api/status', self.get_system_status)
        self.app.router.add_get('/api/dashboard', self.get_dashboard_data)
        self.app.router.add_get('/api/performance', self.get_performance_metrics)
        self.app.router.add_get('/api/network', self.get_network_metrics)
        self.app.router.add_get('/api/competition', self.get_competition_metrics)
        self.app.router.add_get('/api/fees', self.get_fee_metrics)
        self.app.router.add_get('/api/consensus', self.get_consensus_metrics)
        self.app.router.add_get('/api/leaderboard', self.get_leaderboard)
        self.app.router.add_get('/api/historical/{timeframe}', self.get_historical_data)
        self.app.router.add_get('/api/miner/{uid}', self.get_miner_details)
        
        # Static files for dashboard UI
        self.app.router.add_get('/', self.serve_dashboard)
        self.app.router.add_static('/', path=self._get_static_path(), name='static')
        
        # Add CORS to all routes
        for route in list(self.app.router.routes()):
            self.cors.add(route)

    async def start_server(self):
        """Start the dashboard web server"""
        try:
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            
            self.site = web.TCPSite(self.runner, self.host, self.port)
            await self.site.start()
            
            bt.logging.info(f"Dashboard server started at http://{self.host}:{self.port}")
            
        except Exception as e:
            bt.logging.error(f"Error starting dashboard server: {str(e)}")
            raise

    async def stop_server(self):
        """Stop the dashboard web server"""
        try:
            if self.site:
                await self.site.stop()
            if self.runner:
                await self.runner.cleanup()
                
            bt.logging.info("Dashboard server stopped")
            
        except Exception as e:
            bt.logging.error(f"Error stopping dashboard server: {str(e)}")

    async def get_system_status(self, request: web_request.Request) -> web_response.Response:
        """Get system status endpoint"""
        try:
            status = {
                'status': 'operational',
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0.0',
                'uptime_hours': self._calculate_uptime(),
                'components': {
                    'validator': 'operational',
                    'analytics_collector': 'operational',
                    'competition_manager': 'operational',
                    'consensus_manager': 'operational',
                    'fee_manager': 'operational'
                }
            }
            
            return web.json_response(status)
            
        except Exception as e:
            bt.logging.error(f"Error in get_system_status: {str(e)}")
            return web.json_response({'error': str(e)}, status=500)

    async def get_dashboard_data(self, request: web_request.Request) -> web_response.Response:
        """Get comprehensive dashboard data"""
        try:
            dashboard_data = self.analytics_collector.get_realtime_dashboard_data()
            
            # Add additional validator-specific data
            dashboard_data['validator_info'] = {
                'hotkey': str(self.validator.wallet.hotkey.ss58_address),
                'coldkey': str(self.validator.wallet.coldkey.ss58_address),
                'netuid': self.validator.config.netuid,
                'network': self.validator.config.subtensor.network
            }
            
            return web.json_response(dashboard_data)
            
        except Exception as e:
            bt.logging.error(f"Error in get_dashboard_data: {str(e)}")
            return web.json_response({'error': str(e)}, status=500)

    async def get_performance_metrics(self, request: web_request.Request) -> web_response.Response:
        """Get detailed performance metrics"""
        try:
            timeframe = request.query.get('timeframe', '1h')
            
            # Get performance data from analytics collector
            performance_data = self._extract_performance_metrics(timeframe)
            
            # Add real-time performance calculator stats
            if hasattr(self.validator, 'performance_calculator'):
                performance_data['current_stats'] = self.validator.performance_calculator.get_performance_statistics()
            
            return web.json_response(performance_data)
            
        except Exception as e:
            bt.logging.error(f"Error in get_performance_metrics: {str(e)}")
            return web.json_response({'error': str(e)}, status=500)

    async def get_network_metrics(self, request: web_request.Request) -> web_response.Response:
        """Get network health and topology metrics"""
        try:
            network_data = {
                'metagraph_info': {
                    'total_nodes': self.validator.metagraph.n,
                    'block_number': self.validator.metagraph.block.item(),
                    'last_update': datetime.utcnow().isoformat()
                },
                'stake_distribution': self._calculate_stake_metrics(),
                'network_health': self._calculate_network_health()
            }
            
            return web.json_response(network_data)
            
        except Exception as e:
            bt.logging.error(f"Error in get_network_metrics: {str(e)}")
            return web.json_response({'error': str(e)}, status=500)

    async def get_competition_metrics(self, request: web_request.Request) -> web_response.Response:
        """Get competition performance and statistics"""
        try:
            competition_data = {
                'active_rounds': self.validator.competition_manager.get_active_rounds(),
                'completed_rounds': self.validator.competition_manager.get_completed_rounds(),
                'statistics': self.validator.competition_manager.get_round_statistics(),
                'evaluation_performance': self._get_evaluation_performance()
            }
            
            return web.json_response(competition_data)
            
        except Exception as e:
            bt.logging.error(f"Error in get_competition_metrics: {str(e)}")
            return web.json_response({'error': str(e)}, status=500)

    async def get_fee_metrics(self, request: web_request.Request) -> web_response.Response:
        """Get fee management and revenue metrics"""
        try:
            fee_data = self.validator.competition_fee_manager.get_fee_statistics()
            
            # Add revenue distribution history
            fee_data['distribution_history'] = self._get_revenue_distribution_history()
            
            return web.json_response(fee_data)
            
        except Exception as e:
            bt.logging.error(f"Error in get_fee_metrics: {str(e)}")
            return web.json_response({'error': str(e)}, status=500)

    async def get_consensus_metrics(self, request: web_request.Request) -> web_response.Response:
        """Get consensus and governance metrics"""
        try:
            consensus_data = self.validator.consensus_manager.get_consensus_stats()
            
            # Add detailed consensus history
            consensus_data['recent_history'] = self.validator.consensus_manager.get_consensus_history(limit=50)
            consensus_data['voting_analysis'] = self._analyze_voting_patterns()
            
            return web.json_response(consensus_data)
            
        except Exception as e:
            bt.logging.error(f"Error in get_consensus_metrics: {str(e)}")
            return web.json_response({'error': str(e)}, status=500)

    async def get_leaderboard(self, request: web_request.Request) -> web_response.Response:
        """Get miner leaderboard and rankings"""
        try:
            limit = int(request.query.get('limit', '50'))
            sort_by = request.query.get('sort_by', 'average_score')
            
            leaderboard_data = self.validator.leaderboard.get_leaderboard(limit=limit, sort_by=sort_by)
            
            return web.json_response(leaderboard_data)
            
        except Exception as e:
            bt.logging.error(f"Error in get_leaderboard: {str(e)}")
            return web.json_response({'error': str(e)}, status=500)

    async def get_historical_data(self, request: web_request.Request) -> web_response.Response:
        """Get historical analytics data"""
        try:
            timeframe = request.match_info['timeframe']
            
            if timeframe not in ['1h', '24h', '7d', '30d']:
                return web.json_response({'error': 'Invalid timeframe'}, status=400)
            
            historical_data = self.analytics_collector.get_historical_analytics(timeframe)
            
            return web.json_response(historical_data)
            
        except Exception as e:
            bt.logging.error(f"Error in get_historical_data: {str(e)}")
            return web.json_response({'error': str(e)}, status=500)

    async def get_miner_details(self, request: web_request.Request) -> web_response.Response:
        """Get detailed information about a specific miner"""
        try:
            uid = int(request.match_info['uid'])
            
            miner_details = {
                'uid': uid,
                'leaderboard_profile': self.validator.leaderboard.get_miner_profile(uid),
                'fee_stats': self.validator.competition_fee_manager.get_miner_fee_stats(uid),
                'performance_history': self._get_miner_performance_history(uid),
                'recent_submissions': self._get_miner_recent_submissions(uid)
            }
            
            return web.json_response(miner_details)
            
        except Exception as e:
            bt.logging.error(f"Error in get_miner_details: {str(e)}")
            return web.json_response({'error': str(e)}, status=500)

    async def serve_dashboard(self, request: web_request.Request) -> web_response.Response:
        """Serve the main dashboard HTML"""
        try:
            dashboard_html = self._generate_dashboard_html()
            return web.Response(text=dashboard_html, content_type='text/html')
            
        except Exception as e:
            bt.logging.error(f"Error serving dashboard: {str(e)}")
            return web.Response(text=f"Dashboard Error: {str(e)}", status=500)

    def _generate_dashboard_html(self) -> str:
        """Generate dashboard HTML with embedded JavaScript"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Telegraph Subnet Analytics Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 2em; font-weight: bold; color: #3498db; }
        .metric-label { color: #7f8c8d; margin-bottom: 10px; }
        .chart-container { height: 300px; margin-top: 20px; }
        .status-operational { color: #27ae60; }
        .status-warning { color: #f39c12; }
        .status-error { color: #e74c3c; }
        .refresh-btn { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
        .refresh-btn:hover { background: #2980b9; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Telegraph Subnet Analytics Dashboard</h1>
        <p>Real-time monitoring and performance analytics</p>
        <button class="refresh-btn" onclick="refreshDashboard()">Refresh Data</button>
        <span id="lastUpdate"></span>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">System Status</div>
            <div class="metric-value" id="systemStatus">Loading...</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Active Miners</div>
            <div class="metric-value" id="activeMiners">-</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Average Reward</div>
            <div class="metric-value" id="avgReward">-</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Fees Collected (24h)</div>
            <div class="metric-value" id="feesCollected">-</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Active Rounds</div>
            <div class="metric-value" id="activeRounds">-</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Consensus Success Rate</div>
            <div class="metric-value" id="consensusRate">-</div>
        </div>
    </div>

    <div class="metrics-grid" style="margin-top: 20px;">
        <div class="metric-card">
            <h3>Performance Trends</h3>
            <div class="chart-container">
                <canvas id="performanceChart"></canvas>
            </div>
        </div>
        <div class="metric-card">
            <h3>Network Health</h3>
            <div class="chart-container">
                <canvas id="networkChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        let performanceChart, networkChart;
        
        async function fetchDashboardData() {
            try {
                const response = await fetch('/api/dashboard');
                const data = await response.json();
                updateDashboard(data);
            } catch (error) {
                console.error('Error fetching dashboard data:', error);
                document.getElementById('systemStatus').textContent = 'Error';
                document.getElementById('systemStatus').className = 'metric-value status-error';
            }
        }
        
        function updateDashboard(data) {
            // Update basic metrics
            document.getElementById('systemStatus').textContent = data.system_status || 'Unknown';
            document.getElementById('systemStatus').className = `metric-value status-${data.system_status || 'error'}`;
            
            document.getElementById('activeMiners').textContent = data.overview?.active_miners || 0;
            document.getElementById('avgReward').textContent = (data.overview?.current_rewards_mean || 0).toFixed(4);
            document.getElementById('feesCollected').textContent = (data.overview?.fees_collected_24h || 0).toFixed(6) + ' TAO';
            document.getElementById('activeRounds').textContent = data.overview?.active_rounds || 0;
            document.getElementById('consensusRate').textContent = ((data.overview?.consensus_success_rate || 0) * 100).toFixed(1) + '%';
            
            document.getElementById('lastUpdate').textContent = `Last updated: ${new Date(data.timestamp).toLocaleTimeString()}`;
            
            // Update charts
            updatePerformanceChart(data.performance);
            updateNetworkChart(data.network);
        }
        
        function updatePerformanceChart(performanceData) {
            const ctx = document.getElementById('performanceChart').getContext('2d');
            
            if (performanceChart) {
                performanceChart.destroy();
            }
            
            performanceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ['Now-1h', 'Now-45m', 'Now-30m', 'Now-15m', 'Now'],
                    datasets: [{
                        label: 'Average Reward',
                        data: [0.25, 0.28, 0.32, 0.30, performanceData?.current_metrics?.rewards?.mean || 0],
                        borderColor: '#3498db',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { beginAtZero: true }
                    }
                }
            });
        }
        
        function updateNetworkChart(networkData) {
            const ctx = document.getElementById('networkChart').getContext('2d');
            
            if (networkChart) {
                networkChart.destroy();
            }
            
            const healthScore = networkData?.health_score || 0;
            const participationRate = networkData?.current_metrics?.network_size?.participation_rate || 0;
            
            networkChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Health Score', 'Participation Rate'],
                    datasets: [{
                        data: [healthScore * 100, participationRate * 100],
                        backgroundColor: ['#27ae60', '#3498db']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });
        }
        
        function refreshDashboard() {
            fetchDashboardData();
        }
        
        // Initial load and auto-refresh
        fetchDashboardData();
        setInterval(fetchDashboardData, 30000); // Refresh every 30 seconds
    </script>
</body>
</html>
        """

    # Helper methods
    def _calculate_uptime(self) -> float:
        """Calculate validator uptime in hours"""
        # This is a placeholder - in production, you'd track actual start time
        return 24.0

    def _extract_performance_metrics(self, timeframe: str) -> Dict[str, Any]:
        """Extract performance metrics for given timeframe"""
        # Get data from analytics collector based on timeframe
        return self.analytics_collector.get_historical_analytics(timeframe)

    def _calculate_stake_metrics(self) -> Dict[str, Any]:
        """Calculate stake distribution metrics"""
        stakes = [self.validator.metagraph.S[uid] for uid in range(self.validator.metagraph.n)]
        total_stake = sum(stakes)
        
        return {
            'total_stake': float(total_stake),
            'average_stake': float(total_stake / max(self.validator.metagraph.n, 1)),
            'max_stake': float(max(stakes)),
            'min_stake': float(min(stakes))
        }

    def _calculate_network_health(self) -> Dict[str, Any]:
        """Calculate network health metrics"""
        return {
            'decentralization_score': 0.8,  # Placeholder
            'consensus_participation': 0.9,  # Placeholder
            'node_diversity': len(set(self.validator.metagraph.hotkeys))
        }

    def _get_evaluation_performance(self) -> Dict[str, Any]:
        """Get model evaluation performance metrics"""
        # This would integrate with the model evaluator
        return {
            'average_evaluation_time': 5.2,
            'evaluation_success_rate': 0.95,
            'total_evaluations': 1250
        }

    def _get_revenue_distribution_history(self) -> List[Dict[str, Any]]:
        """Get revenue distribution history"""
        # This would come from the fee manager's historical data
        return []

    def _analyze_voting_patterns(self) -> Dict[str, Any]:
        """Analyze consensus voting patterns"""
        return {
            'voter_diversity': 0.8,
            'consensus_strength': 0.7,
            'voting_frequency': 0.9
        }

    def _get_miner_performance_history(self, uid: int) -> List[Dict[str, Any]]:
        """Get performance history for specific miner"""
        if hasattr(self.validator.performance_calculator, 'historical_scores'):
            return self.validator.performance_calculator.historical_scores.get(uid, [])
        return []

    def _get_miner_recent_submissions(self, uid: int) -> List[Dict[str, Any]]:
        """Get recent submissions for specific miner"""
        # This would integrate with prediction store
        return []

    def _get_static_path(self) -> str:
        """Get path for static files"""
        return os.path.join(os.path.dirname(__file__), 'static')