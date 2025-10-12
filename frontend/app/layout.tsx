// USE THIS CODE - THIS IS THE NEW, CORRECTED VERSION
import type { Metadata } from "next";
import { Inter } from "next/font/google"; // 1. Import the Inter font
import "./globals.css";

// 2. Configure the Inter font
const inter = Inter({ subsets: ["latin"] });

// 3. (Optional but recommended) Update your app's metadata
export const metadata: Metadata = {
  title: "Excel Axiom",
  description: "Interactive Excel Filter & Compare Tool",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      {/* 4. Apply the Inter font class to the body */}
      <body className={inter.className}>{children}</body>
    </html>
  );
}