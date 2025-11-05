import { useCallback, useState } from "react";
import { CheckCircleIcon, XMarkIcon, ExclamationTriangleIcon } from "@heroicons/react/24/outline";
import { uxCopy } from "../lib/uxCopy";

export type ToastType = "success" | "error" | "info";

export interface Toast {
  id: string;
  type: ToastType;
  message: string;
  duration?: number; // ms; 0 = no auto-dismiss
}

interface ToastsContextType {
  toasts: Toast[];
  addToast: (type: ToastType, message: string, duration?: number) => void;
  removeToast: (id: string) => void;
  showSuccess: (message: string) => void;
  showError: (message: string) => void;
}

import { createContext, useContext, ReactNode } from "react";

const ToastsContext = createContext<ToastsContextType | undefined>(undefined);

export function ToastsProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const addToast = useCallback(
    (type: ToastType, message: string, duration: number = 5000) => {
      const id = `toast-${Date.now()}-${Math.random()}`;
      setToasts((prev) => [...prev, { id, type, message, duration }]);

      if (duration > 0) {
        setTimeout(() => removeToast(id), duration);
      }
    },
    [removeToast]
  );

  const showSuccess = useCallback(
    (message: string) => addToast("success", message, 3000),
    [addToast]
  );

  const showError = useCallback(
    (message: string) => addToast("error", message, 5000),
    [addToast]
  );

  return (
    <ToastsContext.Provider value={{ toasts, addToast, removeToast, showSuccess, showError }}>
      {children}
      <ToastContainer toasts={toasts} removeToast={removeToast} />
    </ToastsContext.Provider>
  );
}

export function useToasts() {
  const context = useContext(ToastsContext);
  if (!context) {
    throw new Error("useToasts must be used within ToastsProvider");
  }
  return context;
}

/**
 * Toast container that renders all active toasts in a stack
 */
function ToastContainer({
  toasts,
  removeToast,
}: {
  toasts: Toast[];
  removeToast: (id: string) => void;
}) {
  if (toasts.length === 0) return null;

  return (
    <div className="pointer-events-none fixed inset-0 flex flex-col items-end justify-end gap-3 p-4">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onClose={() => removeToast(toast.id)} />
      ))}
    </div>
  );
}

/**
 * Individual toast item
 */
function ToastItem({ toast, onClose }: { toast: Toast; onClose: () => void }) {
  const bgColor = {
    success: "bg-emerald-500/10 border-emerald-500/40",
    error: "bg-red-500/10 border-red-500/40",
    info: "bg-blue-500/10 border-blue-500/40",
  }[toast.type];

  const textColor = {
    success: "text-emerald-100",
    error: "text-red-100",
    info: "text-blue-100",
  }[toast.type];

  const Icon = {
    success: CheckCircleIcon,
    error: ExclamationTriangleIcon,
    info: ExclamationTriangleIcon,
  }[toast.type];

  const iconColor = {
    success: "text-emerald-400",
    error: "text-red-400",
    info: "text-blue-400",
  }[toast.type];

  return (
    <div
      className={`pointer-events-auto flex max-w-sm items-start gap-3 rounded-lg border px-4 py-3 text-sm ${bgColor} ${textColor} shadow-lg backdrop-blur`}
      role="alert"
      aria-live="polite"
    >
      <Icon className={`mt-0.5 h-5 w-5 flex-shrink-0 ${iconColor}`} />
      <p className="flex-1">{toast.message}</p>
      <button
        onClick={onClose}
        className="flex-shrink-0 opacity-70 transition hover:opacity-100"
        aria-label="Close notification"
      >
        <XMarkIcon className="h-4 w-4" />
      </button>
    </div>
  );
}

/**
 * Convenience hook for showing predefined toasts from uxCopy
 */
export function useToastMessages() {
  const { showSuccess, showError } = useToasts();

  return {
    profileSaved: () => showSuccess(uxCopy.toasts.success.profileSaved),
    profileSaveFailed: () => showError(uxCopy.toasts.errors.profileSaveFailed),
    recommendationsReady: () => showSuccess(uxCopy.toasts.success.recommendationsReady),
    recommendationLoadFailed: () => showError(uxCopy.toasts.errors.recommendationLoadFailed),
    networkError: () => showError(uxCopy.toasts.errors.networkError),
    validationError: () => showError(uxCopy.toasts.errors.validationError),
  };
}
