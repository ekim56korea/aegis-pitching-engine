"use client";

import { usePitchStore } from "@/store/pitchStore";
import { Trophy, Target, Activity, Zap, Shield, Loader2, GitCompare } from "lucide-react";
import clsx from "clsx";

const STRATEGY_COLORS = {
  AGGRESSIVE: {
    bg: "bg-red-500/20",
    border: "border-red-500",
    text: "text-red-400",
  },
  CONSERVATIVE: {
    bg: "bg-blue-500/20",
    border: "border-blue-500",
    text: "text-blue-400",
  },
  BALANCED: {
    bg: "bg-gray-500/20",
    border: "border-gray-500",
    text: "text-gray-400",
  },
};

interface CommanderBoardProps {
  onCardClick?: (pitchId: string) => void;
  selectedId?: string | null;
}

export default function CommanderBoard({ onCardClick, selectedId }: CommanderBoardProps) {
  const {
    recommendations,
    bestPitchId,
    scenarioDescription,
    isRecommendationLoading,
    recommendationError,
    comparePitchId,
    toggleCompareMode,
  } = usePitchStore();

  // Helper: Get usage percentage color coding
  const getUsageColor = (percentage: number) => {
    if (percentage > 40) {
      return {
        bg: "bg-emerald-500/20",
        border: "border-emerald-500",
        text: "text-emerald-400",
        barBg: "bg-emerald-500",
      };
    } else if (percentage >= 20) {
      return {
        bg: "bg-blue-500/20",
        border: "border-blue-500",
        text: "text-blue-400",
        barBg: "bg-blue-500",
      };
    } else {
      return {
        bg: "bg-slate-500/20",
        border: "border-slate-500",
        text: "text-slate-400",
        barBg: "bg-slate-500",
      };
    }
  };

  if (isRecommendationLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <Loader2 className="w-8 h-8 text-blue-400 animate-spin mx-auto mb-3" />
          <p className="text-sm text-slate-400">Analyzing pitch options...</p>
        </div>
      </div>
    );
  }

  if (recommendationError) {
    return (
      <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/30">
        <p className="text-sm text-red-400">❌ {recommendationError}</p>
      </div>
    );
  }

  if (recommendations.length === 0) {
    return (
      <div className="p-6 rounded-lg bg-slate-800/50 border border-slate-700 text-center">
        <Trophy className="w-12 h-12 text-slate-600 mx-auto mb-3" />
        <p className="text-sm text-slate-400">
          Configure game context and pitches to see recommendations
        </p>
      </div>
    );
  }

  const primaryPitch = recommendations[0];
  const secondaryPitches = recommendations.slice(1);

  // Helper: Safe access to usage_percentage with fallback
  const getUsagePercentage = (pitch: any): number => {
    return pitch.usage_percentage ?? 0;
  };

  const getStrategyColor = (strategyType: string) => {
    const normalized = strategyType.toUpperCase();
    return STRATEGY_COLORS[normalized as keyof typeof STRATEGY_COLORS] || STRATEGY_COLORS.BALANCED;
  };

  return (
    <div className="space-y-4">
      {/* Scenario Description */}
      {scenarioDescription && (
        <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/30">
          <p className="text-sm text-blue-300">{scenarioDescription}</p>
        </div>
      )}

      {/* Primary Recommendation (Rank 1) */}
      <div
        onClick={() => onCardClick?.(primaryPitch.pitch_id)}
        className={clsx(
          "relative transition-all duration-300 cursor-pointer",
          selectedId === primaryPitch.pitch_id ? "scale-[1.02]" : "hover:scale-[1.01]"
        )}
      >
        <div className="absolute -top-2 -left-2 z-10">
          <div className="bg-yellow-500 text-slate-900 font-bold text-xs px-2 py-1 rounded-full">
            #1 BEST
          </div>
        </div>
        {/* VS Button (Top-Right) - Disabled for primary */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            console.warn('Cannot compare primary pitch to itself');
          }}
          disabled
          className="absolute -top-2 -right-2 z-10 p-2 rounded-full bg-slate-700 border-2 border-slate-600 opacity-50 cursor-not-allowed"
          title="Cannot compare primary to itself"
        >
          <GitCompare className="w-3 h-3 text-slate-500" />
        </button>
        <div className={clsx(
          "p-6 rounded-xl border-4 bg-gradient-to-br from-yellow-500/10 via-slate-900/90 to-slate-900",
          selectedId === primaryPitch.pitch_id
            ? "border-yellow-400 shadow-xl shadow-yellow-500/50"
            : "border-yellow-500"
        )}>
          {/* Header */}
          <div className="flex items-start justify-between mb-4">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <Trophy className="w-5 h-5 text-yellow-500" />
                <h3 className="text-2xl font-bold text-white">
                  {primaryPitch.pitch_type}
                </h3>
              </div>
              {/* Usage Percentage Badge */}
              <div
                className={clsx(
                  "inline-flex items-center gap-1 px-2 py-1 rounded-full border-2 text-xs font-bold",
                  getUsageColor(getUsagePercentage(primaryPitch)).bg,
                  getUsageColor(getUsagePercentage(primaryPitch)).border,
                  getUsageColor(getUsagePercentage(primaryPitch)).text
                )}
              >
                <span>Rec: {getUsagePercentage(primaryPitch).toFixed(1)}%</span>
              </div>
              {/* Progress Bar */}
              <div className="mt-2 w-32 h-2 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className={clsx(
                    "h-full transition-all duration-500",
                    getUsageColor(getUsagePercentage(primaryPitch)).barBg
                  )}
                  style={{ width: `${Math.min(getUsagePercentage(primaryPitch), 100)}%` }}
                />
              </div>
            </div>
            {/* Strategy Badge */}
            <div
              className={clsx(
                "px-3 py-1 rounded-full border-2 text-xs font-bold uppercase",
                getStrategyColor(primaryPitch.strategy_type).bg,
                getStrategyColor(primaryPitch.strategy_type).border,
                getStrategyColor(primaryPitch.strategy_type).text
              )}
            >
              {primaryPitch.strategy_type}
            </div>
          </div>

          {/* Metrics Grid */}
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700">
              <div className="flex items-center gap-2 mb-1">
                <Zap className="w-3 h-3 text-yellow-500" />
                <span className="text-xs text-slate-400">xRV</span>
              </div>
              <div className="text-lg font-bold text-white">
                {primaryPitch.xrv.toFixed(3)}
              </div>
            </div>
            <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700">
              <div className="flex items-center gap-2 mb-1">
                <Shield className="w-3 h-3 text-blue-400" />
                <span className="text-xs text-slate-400">VaR 95%</span>
              </div>
              <div className="text-lg font-bold text-white">
                {primaryPitch.var_95.toFixed(3)}
              </div>
            </div>
            <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700">
              <div className="flex items-center gap-2 mb-1">
                <Target className="w-3 h-3 text-green-400" />
                <span className="text-xs text-slate-400">Strike Rate</span>
              </div>
              <div className="text-lg font-bold text-white">
                {(primaryPitch.strike_rate * 100).toFixed(1)}%
              </div>
            </div>
            <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700">
              <div className="flex items-center gap-2 mb-1">
                <Activity className="w-3 h-3 text-purple-400" />
                <span className="text-xs text-slate-400">Score</span>
              </div>
              <div className="text-lg font-bold text-white">
                {primaryPitch.intelligence_score.toFixed(3)}
              </div>
            </div>
          </div>

          {/* Rationale */}
          <div className="p-3 rounded-lg bg-slate-800/30 border border-slate-700">
            <p className="text-sm text-slate-300 leading-relaxed">
              {primaryPitch.strategy_rationale}
            </p>
          </div>

          {/* Data Quality Indicator */}
          {!primaryPitch.is_reliable && (
            <div className="mt-3 p-2 rounded bg-yellow-500/10 border border-yellow-500/30">
              <p className="text-xs text-yellow-400">
                ⚠️ Low data quality (score: {primaryPitch.data_quality.toFixed(2)})
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Secondary Recommendations (Rank 2+) */}
      {secondaryPitches.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {secondaryPitches.map((pitch) => {
            const strategyColor = getStrategyColor(pitch.strategy_type);
            return (
              <div
                key={pitch.pitch_id}
                onClick={() => onCardClick?.(pitch.pitch_id)}
                className={clsx(
                  "relative p-4 rounded-lg border-2 bg-slate-900/50 transition-all cursor-pointer",
                  selectedId === pitch.pitch_id
                    ? "border-blue-400 shadow-lg shadow-blue-500/30 scale-[1.02]"
                    : "border-slate-700 hover:border-slate-600"
                )}
              >
                {/* VS Button (Top-Right) */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleCompareMode?.(pitch.pitch_id);
                  }}
                  className={clsx(
                    "absolute -top-2 -right-2 z-10 p-2 rounded-full border-2 transition-all",
                    comparePitchId === pitch.pitch_id
                      ? "bg-purple-600 border-purple-400 shadow-lg shadow-purple-500/50"
                      : "bg-slate-800 border-slate-600 hover:bg-slate-700 hover:border-slate-500"
                  )}
                  title={comparePitchId === pitch.pitch_id ? "Remove from comparison" : "Compare with primary"}
                >
                  <GitCompare className={clsx(
                    "w-3 h-3",
                    comparePitchId === pitch.pitch_id ? "text-white" : "text-slate-400"
                  )} />
                </button>
                {/* Header */}
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs font-bold text-slate-500">
                        #{pitch.rank}
                      </span>
                      <h4 className="text-lg font-bold text-white">
                        {pitch.pitch_type}
                      </h4>
                    </div>
                    {/* Usage Percentage Badge */}
                    <div
                      className={clsx(
                        "inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-[10px] font-bold",
                        getUsageColor(getUsagePercentage(pitch)).bg,
                        getUsageColor(getUsagePercentage(pitch)).border,
                        getUsageColor(getUsagePercentage(pitch)).text
                      )}
                    >
                      <span>Rec: {getUsagePercentage(pitch).toFixed(1)}%</span>
                    </div>
                    {/* Progress Bar */}
                    <div className="mt-1.5 w-24 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className={clsx(
                          "h-full transition-all duration-500",
                          getUsageColor(getUsagePercentage(pitch)).barBg
                        )}
                        style={{ width: `${Math.min(getUsagePercentage(pitch), 100)}%` }}
                      />
                    </div>
                  </div>
                  {/* Strategy Badge */}
                  <div
                    className={clsx(
                      "px-2 py-0.5 rounded-full border text-[10px] font-bold uppercase",
                      strategyColor.bg,
                      strategyColor.border,
                      strategyColor.text
                    )}
                  >
                    {pitch.strategy_type}
                  </div>
                </div>

                {/* Metrics */}
                <div className="grid grid-cols-2 gap-2 mb-3">
                  <div>
                    <div className="text-[10px] text-slate-500 uppercase">xRV</div>
                    <div className="text-sm font-bold text-white">
                      {pitch.xrv.toFixed(3)}
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-500 uppercase">VaR</div>
                    <div className="text-sm font-bold text-white">
                      {pitch.var_95.toFixed(3)}
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-500 uppercase">Strike</div>
                    <div className="text-sm font-bold text-white">
                      {(pitch.strike_rate * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-500 uppercase">Score</div>
                    <div className="text-sm font-bold text-white">
                      {pitch.intelligence_score.toFixed(3)}
                    </div>
                  </div>
                </div>

                {/* Mini Rationale */}
                <p className="text-xs text-slate-400 line-clamp-2">
                  {pitch.strategy_rationale}
                </p>

                {/* Data Quality Warning */}
                {!pitch.is_reliable && (
                  <div className="mt-2 text-[10px] text-yellow-500">
                    ⚠️ Low confidence
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
