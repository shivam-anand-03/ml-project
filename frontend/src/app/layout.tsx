import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";
import { Sidebar } from "@/components/nav";

export const metadata: Metadata = {
  title: "AI TriboPredict",
  description: "Data-driven prediction of lubricant friction and wear using machine learning — an experimental and ML approach to data-enabled tribology.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Providers>
          <div className="flex min-h-screen flex-col lg:flex-row">
            <Sidebar />
            <main className="min-w-0 flex-1 p-4 sm:p-6 lg:p-8">{children}</main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
