import { Building2, ArrowUpDown } from "lucide-react";
import { useDashboard } from "../context/DashboardContext";

export function IndustryComparisonTable() {
  const { currentRecord, currentEntityId, comparisonRecords, loadingComparisons } = useDashboard();

  // Combine current record with comparison records for the table
  const allCompanies = currentRecord && currentEntityId 
    ? [
        { ...currentRecord, isCurrent: true },
        ...comparisonRecords.map(record => ({ ...record, isCurrent: false }))
      ]
    : [];

  // Sort by score (lower is better)
  const sortedCompanies = [...allCompanies].sort((a, b) => a.overall_score - b.overall_score);

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 h-full">
      <div className="p-6 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-slate-500 to-slate-600 flex items-center justify-center">
            <Building2 className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-slate-900">Industry Comparison</h2>
            <p className="text-slate-500 text-sm">
              {currentRecord?.region_name || "Loading..."} • {currentRecord?.country_name || ""}
            </p>
          </div>
        </div>
      </div>

      {loadingComparisons && (
        <div className="p-6 text-center text-slate-500">
          Loading comparisons...
        </div>
      )}

      {!loadingComparisons && sortedCompanies.length === 0 && (
        <div className="p-6 text-center text-slate-500">
          No comparison data available
        </div>
      )}

      {!loadingComparisons && sortedCompanies.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-100">
                <th className="text-left px-6 py-3 text-slate-600 text-sm">
                  <div className="flex items-center gap-1">
                    Entity ID
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th className="text-left px-6 py-3 text-slate-600 text-sm">Country</th>
                <th className="text-right px-6 py-3 text-slate-600 text-sm">
                  <div className="flex items-center justify-end gap-1">
                    Score
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th className="text-right px-6 py-3 text-slate-600 text-sm">
                  <div className="flex items-center justify-end gap-1">
                    Scope 1 (tCO₂e)
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th className="text-right px-6 py-3 text-slate-600 text-sm">
                  <div className="flex items-center justify-end gap-1">
                    Scope 2 (tCO₂e)
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th className="text-right px-6 py-3 text-slate-600 text-sm">
                  <div className="flex items-center justify-end gap-1">
                    Revenue (M)
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedCompanies.map((company, index) => {
                const isCurrentCompany = company.isCurrent;
                return (
                  <tr
                    key={company.entity_id}
                    className={`border-b border-slate-50 transition-colors hover:bg-slate-50 ${
                      isCurrentCompany ? "bg-blue-50" : ""
                    }`}
                  >
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        <span className="text-slate-900">
                          {company.entity_id}
                        </span>
                        {isCurrentCompany && (
                          <span className="px-2 py-0.5 bg-blue-600 text-white text-xs rounded">
                            You
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-slate-900">{company.country_name}</span>
                    </td>
                    <td className="px-6 py-4 text-right">
                      <span
                        className={`inline-flex items-center justify-center w-12 h-7 rounded ${
                          company.overall_score <= 2.5
                            ? "bg-emerald-100 text-emerald-700"
                            : company.overall_score <= 3.5
                            ? "bg-yellow-100 text-yellow-700"
                            : "bg-red-100 text-red-700"
                        }`}
                      >
                        {company.overall_score.toFixed(2)}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-right text-slate-900">
                      {company.target_scope_1.toLocaleString()}
                    </td>
                    <td className="px-6 py-4 text-right text-slate-900">
                      {company.target_scope_2.toLocaleString()}
                    </td>
                    <td className="px-6 py-4 text-right text-slate-900">
                      ${(company.revenue / 1000000).toFixed(1)}M
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
