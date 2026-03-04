"use client";

import Nav from "@/components/nav";
import Waitlist from "@/components/waitlist";

const ACCENT = "#F97316";
const HUB_URL = "https://specialized-model-startups.vercel.app";


function SectionLabel({ label }: { label: string }) {
  return (
    <div className="reveal flex items-center gap-5 mb-12">
      <span className="text-xs font-semibold uppercase tracking-[0.18em] text-gray-400 shrink-0">{label}</span>
      <div className="flex-1 h-px bg-gray-100" />
    </div>
  );
}

export default function Home() {
  return (
    <div className="min-h-screen bg-white text-[#0a0a0a] overflow-x-hidden">
      <Nav />

      {/* Hero */}
      <section className="relative min-h-screen flex flex-col justify-center px-6 pt-14 overflow-hidden">
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            backgroundImage: `radial-gradient(circle at 20% 30%, ${ACCENT}07 0%, transparent 50%), radial-gradient(circle at 80% 70%, ${ACCENT}05 0%, transparent 50%)`,
          }}
        />

        <div className="relative max-w-5xl mx-auto w-full py-20">
          <div className="fade-up delay-0 mb-8">
            <span
              className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-semibold border"
              style={{ color: ACCENT, borderColor: `${ACCENT}30`, backgroundColor: `${ACCENT}08` }}
            >
              <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ backgroundColor: ACCENT }} />
              Training &middot; Incident Response &middot; ETA Q2 2027
            </span>
          </div>

          <h1 className="fade-up delay-1 text-[clamp(3rem,9vw,6.5rem)] font-bold leading-[0.92] tracking-tight mb-6">
            <span className="serif font-light italic" style={{ color: ACCENT }}>Oncall</span>
            <span>Compass</span>
          </h1>

          <p className="fade-up delay-2 serif text-[clamp(1.25rem,3vw,2rem)] font-light text-gray-500 mb-4 max-w-xl">
            Incidents investigated. Root cause found.
          </p>

          <p className="fade-up delay-3 text-sm text-gray-400 leading-relaxed max-w-lg mb-10">
            First model trained on time-to-correct-root-cause&nbsp;&mdash; ranks hypotheses like the senior engineer who has seen this before, not like a search engine.
          </p>

          <div className="fade-up delay-4">
            <Waitlist />
          </div>
        </div>
      </section>

      {/* The Problem */}
      <section className="px-6 py-24 max-w-5xl mx-auto">
        <SectionLabel label="The Problem" />
        <div className="grid md:grid-cols-2 gap-6">
          <div className="reveal rounded-2xl border border-gray-100 p-8 bg-gray-50/50">
            <p className="text-xs font-semibold uppercase tracking-widest text-gray-400 mb-5">What general models do</p>
            <ul className="space-y-3 text-sm text-gray-500 leading-relaxed">
              <li className="flex gap-3">
                <span className="text-gray-300 mt-0.5">&#8212;</span>
                List every possible cause with no ranking&nbsp;&mdash; at 3am with a P0, an unranked list is just noise
              </li>
              <li className="flex gap-3">
                <span className="text-gray-300 mt-0.5">&#8212;</span>
                Can&apos;t distinguish a P1 from noise&nbsp;&mdash; no understanding of signal priority or blast radius
              </li>
              <li className="flex gap-3">
                <span className="text-gray-300 mt-0.5">&#8212;</span>
                No memory of past incidents&nbsp;&mdash; every investigation starts from scratch with no pattern matching
              </li>
              <li className="flex gap-3">
                <span className="text-gray-300 mt-0.5">&#8212;</span>
                Postmortems that don&apos;t prevent recurrence&nbsp;&mdash; generic recommendations, no system-layer specificity
              </li>
            </ul>
          </div>

          <div
            className="reveal rounded-2xl border p-8"
            style={{ borderColor: `${ACCENT}25`, backgroundColor: `${ACCENT}05` }}
          >
            <p className="text-xs font-semibold uppercase tracking-widest mb-5" style={{ color: ACCENT }}>What OncallCompass does</p>
            <ul className="space-y-3 text-sm leading-relaxed text-gray-700">
              <li className="flex gap-3">
                <svg className="mt-0.5 shrink-0" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                Ranks hypotheses by likelihood given your stack&nbsp;&mdash; the most probable root cause surfaces first
              </li>
              <li className="flex gap-3">
                <svg className="mt-0.5 shrink-0" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                Sequences investigation steps in the order expert SREs actually follow them
              </li>
              <li className="flex gap-3">
                <svg className="mt-0.5 shrink-0" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                Trained on 300k incident-to-resolution traces&nbsp;&mdash; reward signal is drill MTTR reduction
              </li>
              <li className="flex gap-3">
                <svg className="mt-0.5 shrink-0" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                Generates drift-resistant postmortems with prevention-focused, layer-specific action items
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* How It&apos;s Built */}
      <section className="px-6 py-24 bg-gray-50/50">
        <div className="max-w-5xl mx-auto">
          <SectionLabel label="How it&apos;s built" />
          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                step: "01",
                title: "Supervised Fine-Tuning",
                desc: "300k runbooks, postmortems, and incident tickets from GitHub, public engineering blogs, and status page histories. Each training example is an (alert signals + context, ranked hypotheses, resolution trace) triple. OncallCompass learns to read an incident the way a senior SRE reads it.",
              },
              {
                step: "02",
                title: "RL with MTTR Reward",
                desc: "Reward signal: time-to-correct-root-cause in simulated incident drills plus measured MTTR reduction. The model is punished for correct-but-slow diagnoses and for surfacing the right root cause fifth instead of first. Faster, more accurate triaging is the only path to higher reward.",
              },
              {
                step: "03",
                title: "DPO Alignment",
                desc: "Direct Preference Optimization on (fast-resolution path, slow-resolution path) pairs extracted from the training corpus. OncallCompass learns to prefer hypothesis ranking over hypothesis listing, topology-aware reasoning over metric correlation, and prevention-focused postmortems over descriptive ones.",
              },
            ].map(({ step, title, desc }) => {
              // eslint-disable-next-line react-hooks/rules-of-hooks
              return (
                <div key={step} className="reveal-scale rounded-2xl border border-gray-100 bg-white p-8">
                  <div className="text-xs font-bold uppercase tracking-widest mb-4" style={{ color: ACCENT }}>{step}</div>
                  <h3 className="serif font-semibold text-lg mb-3 text-gray-900">{title}</h3>
                  <p className="text-sm text-gray-500 leading-relaxed">{desc}</p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Capabilities */}
      <section className="px-6 py-24 max-w-5xl mx-auto">
        <SectionLabel label="Capabilities" />
        <div className="grid sm:grid-cols-2 gap-5">
          {[
            {
              icon: (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
                </svg>
              ),
              title: "Hypothesis ranking with confidence scores",
              desc: "Outputs an ordered list of root cause candidates with probability weights. Not a dump of possibilities&nbsp;&mdash; a ranked queue with the evidence chain for each hypothesis, so you know exactly what to check first and why.",
            },
            {
              icon: (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/>
                </svg>
              ),
              title: "Step-by-step investigation sequencing",
              desc: "Generates an ordered runbook for the current incident&nbsp;&mdash; not generic steps from a template, but a sequenced investigation path derived from the specific alert signals, stack context, and similar resolved incidents.",
            },
            {
              icon: (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/>
                </svg>
              ),
              title: "Postmortem generation with prevention focus",
              desc: "Generates structured postmortems with timeline reconstruction, contributing factors, and actionable follow-up items. Each action item is layer-specific and measurable&nbsp;&mdash; no generic &ldquo;improve monitoring&rdquo; recommendations that don&apos;t prevent recurrence.",
            },
            {
              icon: (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>
                </svg>
              ),
              title: "Runbook synthesis",
              desc: "Synthesizes incident-specific runbooks from the resolution trace corpus. Detects when an existing runbook is stale or drifted from the current stack and flags it before you waste time following steps that no longer apply.",
            },
          ].map(({ icon, title, desc }) => {
            // eslint-disable-next-line react-hooks/rules-of-hooks
            return (
              <div
                key={title}
               
                className="reveal rounded-2xl border border-gray-100 p-7 flex gap-5 hover:border-gray-200 transition-colors"
              >
                <div
                  className="shrink-0 w-10 h-10 rounded-xl flex items-center justify-center"
                  style={{ backgroundColor: `${ACCENT}10` }}
                >
                  {icon}
                </div>
                <div>
                  <h3 className="font-semibold text-sm text-gray-900 mb-1.5">{title}</h3>
                  <p className="text-sm text-gray-500 leading-relaxed" dangerouslySetInnerHTML={{ __html: desc }} />
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* The Numbers */}
      <section className="px-6 py-24 bg-gray-50/50">
        <div className="max-w-5xl mx-auto">
          <SectionLabel label="The numbers" />
          <div className="grid sm:grid-cols-3 gap-6">
            {[
              { stat: "300k", label: "Incident-to-resolution traces", sub: "Runbooks + postmortems + incident tickets from public sources" },
              { stat: "Qwen2.5-7B", label: "Base model", sub: "Coder-Instruct" },
              { stat: "Drill MTTR", label: "Reward signal", sub: "Time-to-correct-root-cause reduction in simulated incident drills" },
            ].map(({ stat, label, sub }) => {
              // eslint-disable-next-line react-hooks/rules-of-hooks
              return (
                <div
                  key={label}
                 
                  className="reveal rounded-2xl border p-8"
                  style={{ borderColor: `${ACCENT}20` }}
                >
                  <div className="text-3xl font-bold tracking-tight mb-2" style={{ color: ACCENT }}>{stat}</div>
                  <div className="text-sm font-semibold text-gray-800 mb-1">{label}</div>
                  <div className="text-xs text-gray-400">{sub}</div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="px-6 py-12 border-t border-gray-100">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-gray-400">
          <p>
            Part of the{" "}
            <a href={HUB_URL} className="underline underline-offset-2 hover:text-gray-600 transition-colors">
              Specialist AI
            </a>{" "}
            portfolio &middot; Caleb Newton &middot; USC &middot; 2026
          </p>
        </div>
      </footer>
    </div>
  );
}
