import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, Navigate, RouterProvider } from "react-router-dom";
import App from "./App";
import "./index.css";
import OnboardingPage from "./pages/OnboardingPage";
import RecommendationsPage from "./pages/RecommendationsPage";
import ProgressPage from "./pages/ProgressPage";
import ChatPage from "./pages/ChatPage";
import AdminPage from "./pages/AdminPage";

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    children: [
      { index: true, element: <Navigate to="/onboarding" replace /> },
      { path: "onboarding", element: <OnboardingPage /> },
      { path: "recommendations", element: <RecommendationsPage /> },
      { path: "progress", element: <ProgressPage /> },
      { path: "chat", element: <ChatPage /> },
      { path: "admin", element: <AdminPage /> }
    ]
  }
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);
