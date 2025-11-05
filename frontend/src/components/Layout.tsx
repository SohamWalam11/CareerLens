import { NavLink, Outlet } from "react-router-dom";

type NavItem = {
  to: string;
  label: string;
};

const navItems: NavItem[] = [
  { to: "/onboarding", label: "Onboarding" },
  { to: "/recommendations", label: "Recommendations" },
  { to: "/progress", label: "Progress" },
  { to: "/chat", label: "AI Chat" },
  { to: "/admin", label: "Admin" }
];

const Layout = () => {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex">
      <aside className="w-64 border-r border-slate-800 bg-slate-900/70 backdrop-blur">
        <div className="px-6 py-8">
          <h1 className="text-2xl font-semibold text-neon-pink">CareerLens 2.0</h1>
          <p className="mt-2 text-sm text-slate-400">
            Personalized education and career path intelligence.
          </p>
        </div>
        <nav className="mt-4 space-y-1 px-2">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }: { isActive: boolean }) =>
                `block rounded-md px-4 py-2 text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-neon-blue/20 text-neon-blue"
                    : "text-slate-300 hover:bg-slate-800 hover:text-neon-purple"
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-5xl px-8 py-10">
          <Outlet />
        </div>
      </main>
    </div>
  );
};

export default Layout;
