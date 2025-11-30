import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";
import { BarChart3, TrendingDown } from "lucide-react";

export function EmissionsCharts() {
  // Historical emissions data
  const historicalData = [
    { year: "2020", scope1: 14800, scope2: 10200, total: 25000 },
    { year: "2021", scope1: 14200, scope2: 9800, total: 24000 },
    { year: "2022", scope1: 13500, scope2: 9100, total: 22600 },
    { year: "2023", scope1: 13000, scope2: 8600, total: 21600 },
    { year: "2024", scope1: 12500, scope2: 8200, total: 20700 },
  ];

  // Emissions by source
  const emissionsBySource = [
    { name: "Manufacturing Processes", value: 7200, percentage: 35 },
    { name: "Purchased Electricity", value: 5800, percentage: 28 },
    { name: "Fleet & Logistics", value: 3100, percentage: 15 },
    { name: "Heating & Cooling", value: 2400, percentage: 12 },
    { name: "Other Operations", value: 2200, percentage: 10 },
  ];

  // Sector benchmark data
  const sectorBenchmark = [
    { sector: "Chemicals", avgEmissions: 28500 },
    { sector: "Metals", avgEmissions: 35200 },
    { sector: "Food Products", avgEmissions: 18900 },
    { sector: "Your Company", avgEmissions: 20700 },
    { sector: "Textiles", avgEmissions: 16200 },
    { sector: "Electronics", avgEmissions: 14800 },
  ];

  const COLORS = ["#3b82f6", "#8b5cf6", "#ec4899", "#f59e0b", "#10b981"];

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white border border-slate-200 rounded-lg shadow-lg p-3">
          <p className="text-slate-900 mb-2">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {entry.name}: {entry.value.toLocaleString()} tCO₂e
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-6">
      {/* Charts Header */}
      <div className="flex items-center gap-2">
        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center">
          <BarChart3 className="w-5 h-5 text-white" />
        </div>
        <div>
          <h2 className="text-slate-900">Emissions Analytics (PROOF OF CONCEPT) </h2>
          <p className="text-slate-500 text-sm">Comprehensive view of your carbon footprint</p>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Historical Trends */}
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-slate-900">Historical Emissions Trend</h3>
              <p className="text-slate-500 text-sm mt-1">Year-over-year comparison</p>
            </div>
            <div className="flex items-center gap-2 px-3 py-1.5 bg-emerald-50 border border-emerald-200 rounded-lg">
              <TrendingDown className="w-4 h-4 text-emerald-600" />
              <span className="text-emerald-700 text-sm">-17% since 2020</span>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={historicalData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="year" stroke="#64748b" />
              <YAxis stroke="#64748b" />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              <Line
                type="monotone"
                dataKey="scope1"
                stroke="#3b82f6"
                strokeWidth={2}
                name="Scope 1"
                dot={{ fill: "#3b82f6", r: 4 }}
              />
              <Line
                type="monotone"
                dataKey="scope2"
                stroke="#8b5cf6"
                strokeWidth={2}
                name="Scope 2"
                dot={{ fill: "#8b5cf6", r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Emissions by Source */}
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
          <div className="mb-6">
            <h3 className="text-slate-900">Emissions by Source</h3>
            <p className="text-slate-500 text-sm mt-1">Current year breakdown</p>
          </div>
          <div className="flex items-center gap-6">
            <ResponsiveContainer width="50%" height={300}>
              <PieChart>
                <Pie
                  data={emissionsBySource}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {emissionsBySource.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div className="flex-1 space-y-3">
              {emissionsBySource.map((source, index) => (
                <div key={index}>
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-3 h-3 rounded-sm"
                        style={{ backgroundColor: COLORS[index % COLORS.length] }}
                      />
                      <span className="text-slate-700 text-sm">{source.name}</span>
                    </div>
                    <span className="text-slate-900 text-sm">{source.percentage}%</span>
                  </div>
                  <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${source.percentage}%`,
                        backgroundColor: COLORS[index % COLORS.length]
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Sector Benchmark */}
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 lg:col-span-2">
          <div className="mb-6">
            <h3 className="text-slate-900">Industry Sector Benchmark</h3>
            <p className="text-slate-500 text-sm mt-1">Average total emissions by manufacturing sector</p>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={sectorBenchmark} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis type="number" stroke="#64748b" />
              <YAxis dataKey="sector" type="category" stroke="#64748b" width={120} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="avgEmissions" name="Avg Emissions" radius={[0, 8, 8, 0]}>
                {sectorBenchmark.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.sector === "Your Company" ? "#8b5cf6" : "#3b82f6"}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
