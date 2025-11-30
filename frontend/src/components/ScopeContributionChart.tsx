import React from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { BarChart3 } from "lucide-react";

export function ScopeContributionChart() {
  // Scope 1 contributing metrics (direct emissions)
  const scope1Metrics = [
    { metric: "Manufacturing Processes", value: 5200, percentage: 41.6 },
    { metric: "Fleet & Vehicle Operations", value: 3100, percentage: 24.8 },
    { metric: "On-site Combustion", value: 2400, percentage: 19.2 },
    { metric: "Refrigerants & Chemicals", value: 1200, percentage: 9.6 },
    { metric: "Other Direct Sources", value: 600, percentage: 4.8 },
  ];

  // Scope 2 contributing metrics (indirect emissions from purchased energy)
  const scope2Metrics = [
    { metric: "Purchased Electricity", value: 4800, percentage: 58.5 },
    { metric: "Purchased Steam", value: 1800, percentage: 22.0 },
    { metric: "Purchased Heating", value: 1000, percentage: 12.2 },
    { metric: "Purchased Cooling", value: 600, percentage: 7.3 },
  ];

  const scope1Color = "#3b82f6"; // Blue for Scope 1
  const scope2Color = "#8b5cf6"; // Purple for Scope 2

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white border border-slate-200 rounded-lg shadow-lg p-3">
          <p className="text-slate-900 font-medium mb-2">{label}</p>
          {payload.map((entry: any, index: number) => (
            <div key={index} className="space-y-1">
              <p className="text-sm" style={{ color: entry.color }}>
                {entry.name}: {entry.value.toLocaleString()} tCO₂e
              </p>
              {data.percentage && (
                <p className="text-xs text-slate-500">
                  {data.percentage.toFixed(1)}% of total
                </p>
              )}
            </div>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
      <div className="flex items-center gap-2 mb-6">
        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
          <BarChart3 className="w-5 h-5 text-white" />
        </div>
        <div>
          <h3 className="text-slate-900">Scope Contribution Analysis (PROOF OF CONCEPT) </h3>
          <p className="text-slate-500 text-sm mt-1">Top contributing metrics for Scope 1 and Scope 2 emissions</p>
        </div>
      </div>

      {/* Scope 1 Chart */}
      <div className="mb-12 pb-8 border-b border-slate-200">
        <div className="mb-4">
          <h4 className="text-slate-900 font-medium">Scope 1 - Direct Emissions</h4>
          <p className="text-slate-500 text-sm mt-1">Total: 12,500 tCO₂e</p>
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={scope1Metrics} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis type="number" stroke="#64748b" />
            <YAxis dataKey="metric" type="category" stroke="#64748b" width={180} />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="value" name="Scope 1" radius={[0, 8, 8, 0]}>
              {scope1Metrics.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={scope1Color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="mt-4 space-y-2">
          {scope1Metrics.map((metric, index) => (
            <div key={index} className="flex items-center justify-between">
              <div className="flex items-center gap-2 flex-1">
                <div
                  className="w-3 h-3 rounded-sm"
                  style={{ backgroundColor: scope1Color }}
                />
                <span className="text-slate-700 text-sm">{metric.metric}</span>
              </div>
              <div className="flex items-center gap-4">
                <span className="text-slate-900 text-sm font-medium">{metric.value.toLocaleString()} tCO₂e</span>
                <span className="text-slate-500 text-sm w-16 text-right">{metric.percentage.toFixed(1)}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Scope 2 Chart */}
      <div className="mt-12">
        <div className="mb-4">
          <h4 className="text-slate-900 font-medium">Scope 2 - Indirect Emissions (Purchased Energy)</h4>
          <p className="text-slate-500 text-sm mt-1">Total: 8,200 tCO₂e</p>
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={scope2Metrics} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis type="number" stroke="#64748b" />
            <YAxis dataKey="metric" type="category" stroke="#64748b" width={180} />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="value" name="Scope 2" radius={[0, 8, 8, 0]}>
              {scope2Metrics.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={scope2Color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="mt-4 space-y-2">
          {scope2Metrics.map((metric, index) => (
            <div key={index} className="flex items-center justify-between">
              <div className="flex items-center gap-2 flex-1">
                <div
                  className="w-3 h-3 rounded-sm"
                  style={{ backgroundColor: scope2Color }}
                />
                <span className="text-slate-700 text-sm">{metric.metric}</span>
              </div>
              <div className="flex items-center gap-4">
                <span className="text-slate-900 text-sm font-medium">{metric.value.toLocaleString()} tCO₂e</span>
                <span className="text-slate-500 text-sm w-16 text-right">{metric.percentage.toFixed(1)}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

