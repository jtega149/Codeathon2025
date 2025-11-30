import { useState, useRef, useEffect } from "react";
import { ChevronDown, Building2 } from "lucide-react";
import { useDashboard } from "../context/DashboardContext";

export function EntityIdSelector() {
  const { entityIds, currentEntityId, setCurrentEntityId, loading } = useDashboard();
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const handleSelect = (entityId: number) => {
    setCurrentEntityId(entityId);
    setIsOpen(false);
  };

  if (loading || entityIds.length === 0) {
    return (
      <div className="relative">
        <button
          disabled
          className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-300 rounded-lg text-slate-500 cursor-not-allowed"
        >
          <Building2 className="w-4 h-4" />
          <span>Loading...</span>
          <ChevronDown className="w-4 h-4" />
        </button>
      </div>
    );
  }

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-300 rounded-lg hover:border-slate-400 hover:bg-slate-50 transition-colors shadow-sm"
      >
        <Building2 className="w-4 h-4 text-slate-600" />
        <span className="text-slate-900 font-medium">
          Entity ID: {currentEntityId || "N/A"}
        </span>
        <ChevronDown
          className={`w-4 h-4 text-slate-600 transition-transform ${
            isOpen ? "rotate-180" : ""
          }`}
        />
      </button>

      {isOpen && (
        <div className="absolute top-full left-0 mt-2 w-64 bg-white border border-slate-200 rounded-lg shadow-lg z-50 max-h-96 overflow-y-auto">
          <div className="p-2">
            <div className="px-3 py-2 text-xs font-semibold text-slate-500 uppercase tracking-wider border-b border-slate-100">
              Select Entity ID
            </div>
            <div className="py-1">
              {entityIds.map((entityId) => (
                <button
                  key={entityId}
                  onClick={() => handleSelect(entityId)}
                  className={`w-full text-left px-3 py-2 text-sm rounded transition-colors ${
                    entityId === currentEntityId
                      ? "bg-blue-50 text-blue-700 font-medium"
                      : "text-slate-700 hover:bg-slate-50"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span>Entity ID: {entityId}</span>
                    {entityId === currentEntityId && (
                      <span className="text-blue-600">✓</span>
                    )}
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

