import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700", "800"],
});

export const metadata: Metadata = {
  title: "SceneShift — AI Product Placement in Streaming Content",
  description:
    "SceneShift uses multi-agent AI to replace in-scene objects with personalized, era-authentic product placements in real time.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} h-full antialiased dark`}>
      <body
        suppressHydrationWarning
        className="min-h-full flex flex-col bg-[#0a0a0f] text-white"
      >
        {children}
      </body>
    </html>
  );
}
