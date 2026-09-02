import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GROKVERSE — watch a neural network grok",
  description:
    "Reproduce, dissect, and watch grokking happen in real time, in 3D. BWKI 2026.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
