// ============================================================
// Tour.tsx — the first-run tour overlay. A pure overlay in the
// palette's mold: mounting/unmounting it never touches the store,
// the session-resume path, or the persistent GL <Canvas>. All step
// content, ordering and the done-flag rules live headlessly in
// ../tour.ts — this file only measures anchors and renders.
//
// It auto-opens once (tour.ts::shouldShowTour) and can be replayed
// from ⌘K → "Show tour" via the module-scope showTour(), mirroring
// how GraphEditor exposes fitGraphView without a React context.
// ============================================================

import { useCallback, useEffect, useLayoutEffect, useState } from "react";
import {
  TOUR_STEPS,
  anchorSelector,
  isLastStep,
  markTourDone,
  shouldShowTour,
  stepProgress,
  tourReducer,
} from "../tour";
import type { TourAction } from "../tour";
import "./tour.css";

// The live overlay's opener (registered on mount) — module-scope so the ⌘K
// palette can replay the tour without a React context.
let openImpl: (() => void) | null = null;

/** Open (or restart) the tour at step 1. No-op while the overlay is unmounted. */
export function showTour(): void {
  openImpl?.();
}

/** Padded viewport rect of a step's anchor element, or null (centered card). */
function anchorRect(selector: string | null): DOMRect | null {
  if (!selector) return null;
  const el = document.querySelector(selector);
  return el ? el.getBoundingClientRect() : null;
}

export function TourOverlay() {
  // step index, or null = closed. Auto-opens exactly once per profile: the
  // localStorage flag is set on BOTH finish and skip.
  const [step, setStep] = useState<number | null>(() =>
    shouldShowTour(localStorage) ? 0 : null,
  );
  const [rect, setRect] = useState<DOMRect | null>(null);

  useEffect(() => {
    openImpl = () => setStep(0);
    return () => {
      openImpl = null;
    };
  }, []);

  const dispatch = useCallback((action: TourAction) => {
    setStep((s) => {
      const next = tourReducer(s, action);
      // closing — by skip OR by finishing — marks the tour done for good
      if (s !== null && next === null) markTourDone(localStorage);
      return next;
    });
  }, []);

  // measure the current anchor before paint; re-measure on window resize
  // (the mode tabs / toolbar are static chrome, so no observer is needed)
  useLayoutEffect(() => {
    if (step === null) return;
    const measure = () => setRect(anchorRect(anchorSelector(TOUR_STEPS[step])));
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [step]);

  // Esc skips the tour (its own listener: the global keymap owns ⌘K/⌘O/⌘1…
  // and must keep seeing them THROUGH the tour — the app stays fully usable)
  useEffect(() => {
    if (step === null) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") dispatch("skip");
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step, dispatch]);

  if (step === null) return null;
  const s = TOUR_STEPS[step];

  return (
    // pointer-events: none on the root — the tour never blocks the app;
    // only the card (and its buttons) are interactive.
    <div className="tour-root" role="dialog" aria-label="First-run tour">
      {rect && (
        <div
          className="tour-ring"
          style={{
            left: rect.left - 6,
            top: rect.top - 6,
            width: rect.width + 12,
            height: rect.height + 12,
          }}
        />
      )}
      <div className="tour-card">
        <div className="tour-head">
          <span className="tour-progress">{stepProgress(step)}</span>
          <span className="tour-title">{s.title}</span>
        </div>
        <p className="tour-body">{s.body}</p>
        <div className="tour-actions">
          <button className="tour-skip" onClick={() => dispatch("skip")}>
            Skip tour
          </button>
          <span className="tour-spacer" />
          {step > 0 && (
            <button className="tour-back" onClick={() => dispatch("back")}>
              Back
            </button>
          )}
          <button className="tour-next" onClick={() => dispatch("next")} autoFocus>
            {isLastStep(step) ? "Done" : "Next"}
          </button>
        </div>
      </div>
    </div>
  );
}
