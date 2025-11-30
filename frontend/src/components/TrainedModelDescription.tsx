import { Sparkles, Lightbulb, CheckCircle2, AlertCircle, MessageSquare, Send, Loader2 } from "lucide-react";
import { useState, useEffect, React} from "react";
import { useDashboard } from "../context/DashboardContext";
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

interface SubmissionData {
  entity_id: number;
  target_scope_1: number;
  target_scope_2: number;
}

export function TrainedModelDescription() {
  const [data, setData] = useState<SubmissionData[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch("http://localhost:8000/submission-data");
        if (!response.ok) {
          throw new Error("Failed to fetch submission data");
        }
        const result = await response.json();
        setData(result.data || []);
      } catch (error) {
        console.error("Error fetching submission data:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Transform data for scatter plot
  // Create two series: one for target_scope_1 and one for target_scope_2
  const scatterData1 = data.map((item) => ({
    x: item.entity_id,
    y: item.target_scope_1,
  }));

  const scatterData2 = data.map((item) => ({
    x: item.entity_id,
    y: item.target_scope_2,
  }));

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white border border-slate-200 rounded-lg shadow-lg p-3">
          <p className="text-slate-900 mb-2">Entity ID: {payload[0].payload.x}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {entry.name}: {entry.value?.toLocaleString(undefined, { maximumFractionDigits: 2 })} tCO₂e
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200">
      <div className="p-6 border-b border-slate-100">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-slate-900">Our Custom Model</h2>
              <p className="text-slate-500 text-sm">Designing our model</p>
            </div>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 bg-violet-50 border border-violet-200 rounded-lg">
            <AlertCircle className="w-4 h-4 text-violet-600" />
            <span className="text-violet-700 text-sm">Action Required</span>
          </div>
        </div>
      </div>
      <div className="p-6">
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <Loader2 className="w-6 h-6 text-slate-400 animate-spin" />
          </div>
        ) : (
          <div>
            <h3 className="text-slate-900 mb-4">Target Emissions Scatter Plot</h3>
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart
                margin={{ top: 20, right: 20, bottom: 40, left: 80 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis 
                  type="number" 
                  dataKey="x" 
                  name="Entity ID"
                  stroke="#64748b"
                  label={{ value: "Entity ID", position: "insideBottom", offset: -5 }}
                />
                <YAxis 
                  type="number" 
                  dataKey="y" 
                  name="Emissions (tCO₂e)"
                  stroke="#64748b"
                  label={{ value: "Emissions (tCO₂e)", angle: -90, position: "left", offset: 30 }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                <Scatter
                  name="Target Scope 1"
                  data={scatterData1}
                  fill="#3b82f6"
                />
                <Scatter
                  name="Target Scope 2"
                  data={scatterData2}
                  fill="#ef4444"
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
}