import type { Metadata } from "next";
import { Manrope, Source_Serif_4 } from "next/font/google";
import "./globals.css";
import RevealObserver from "@/components/reveal-observer";

const manrope = Manrope({ subsets: ["latin"], variable: "--font-manrope" });
const sourceSerif = Source_Serif_4({
  subsets: ["latin"],
  weight: ["300", "400", "600", "700"],
  variable: "--font-source-serif",
});

export const metadata: Metadata = {
  title: "OncallCompass — Incidents investigated. Root cause found.",
  description:
    "ML-powered incident response sequencing trained on real runbooks, postmortems, and MTTR outcomes. Ranks hypotheses, sequences steps, generates drift-resistant postmortems.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${manrope.variable} ${sourceSerif.variable}`}>
      <body style={{ fontFamily: "var(--font-manrope), system-ui, sans-serif" }}>
        <RevealObserver />{children}
      </body>
    </html>
  );
}
