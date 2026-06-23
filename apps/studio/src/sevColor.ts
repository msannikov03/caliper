// Shared severity / color helpers for the singularity HUD and the tip ellipsoid.

export function severity(sigmaMin: number, epsActivate: number): number {
  // 1 = safe (governor off), 0 = at the singularity.
  return Math.max(0, Math.min(1, sigmaMin / epsActivate));
}

export function rampColor(f: number): string {
  if (f >= 0.66) return "#3ad29f"; // green
  if (f >= 0.33) return "#e0a83a"; // amber
  return "#e5484d"; // red
}
