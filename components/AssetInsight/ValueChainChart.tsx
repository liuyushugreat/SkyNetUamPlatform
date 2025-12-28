import React, { useEffect, useRef } from 'react';
import * as echarts from 'echarts';

const ValueChainChart: React.FC = () => {
  const chartRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!chartRef.current) return;
    const chart = echarts.init(chartRef.current);

    // Data for Graph
    const nodes = [
      { name: 'Edge Nodes\n(UAV/Sensors)', desc: 'Real-time Collection', x: 100, y: 300, symbolSize: 50, itemStyle: { color: '#3b82f6' }, category: 0 },
      { name: 'SkyNet Hub\n(Aggregation)', desc: 'Cleaning & Hashing', x: 400, y: 300, symbolSize: 60, itemStyle: { color: '#8b5cf6' }, category: 1 },
      { name: 'RWA Engine\n(Valuation)', desc: 'Dynamic Pricing', x: 700, y: 300, symbolSize: 60, itemStyle: { color: '#10b981' }, category: 2 },
      { name: 'Data Market\n(Trading)', desc: 'Exchange & Settlement', x: 1000, y: 300, symbolSize: 70, itemStyle: { color: '#f59e0b' }, category: 3 }
    ];

    const links = [
      { source: 'Edge Nodes\n(UAV/Sensors)', target: 'SkyNet Hub\n(Aggregation)' },
      { source: 'SkyNet Hub\n(Aggregation)', target: 'RWA Engine\n(Valuation)' },
      { source: 'RWA Engine\n(Valuation)', target: 'Data Market\n(Trading)' }
    ];

    // Animated Lines Data (Simulating packets flow)
    const linesData = [
      { coords: [[100, 300], [400, 300]] },
      { coords: [[400, 300], [700, 300]] },
      { coords: [[700, 300], [1000, 300]] }
    ];

    const option: echarts.EChartsOption = {
      backgroundColor: 'transparent',
      title: {
        text: 'SkyNet Data Value Chain Topology',
        left: 'center',
        top: 10,
        textStyle: { color: '#e2e8f0', fontSize: 16 }
      },
      tooltip: {},
      xAxis: { show: false, min: 0, max: 1100 },
      yAxis: { show: false, min: 0, max: 600 },
      series: [
        {
          type: 'graph',
          layout: 'none',
          coordinateSystem: 'cartesian2d',
          data: nodes,
          links: links,
          symbol: 'circle',
          label: {
            show: true,
            position: 'bottom',
            formatter: (params: any) => {
              return `${params.name}\n{desc|${params.data.desc}}`;
            },
            rich: {
              desc: {
                color: '#94a3b8',
                fontSize: 10,
                padding: [4, 0, 0, 0],
                fontStyle: 'italic'
              }
            },
            color: '#cbd5e1',
            fontSize: 12,
            lineHeight: 16
          },
          lineStyle: { color: '#475569', width: 2, curveness: 0.1 },
          z: 10
        },
        // Effect Scatter for Nodes (Pulse effect)
        {
          type: 'effectScatter',
          coordinateSystem: 'cartesian2d',
          data: nodes.map(n => [n.x, n.y]),
          symbolSize: 20,
          showEffectOn: 'render',
          rippleEffect: { brushType: 'stroke', scale: 3 },
          itemStyle: { color: '#60a5fa' },
          z: 5
        },
        // Animated Lines (Data Flow)
        {
          type: 'lines',
          coordinateSystem: 'cartesian2d',
          effect: {
            show: true,
            period: 2, // Speed
            trailLength: 0.3,
            color: '#38bdf8',
            symbol: 'arrow',
            symbolSize: 8
          },
          lineStyle: {
            normal: {
              color: '#38bdf8',
              width: 0,
              curveness: 0.1
            }
          },
          data: linesData,
          z: 20
        }
      ]
    };

    chart.setOption(option);

    const handleResize = () => chart.resize();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.dispose();
    };
  }, []);

  return <div ref={chartRef} style={{ width: '100%', height: '250px' }} />;
};

export default ValueChainChart;

