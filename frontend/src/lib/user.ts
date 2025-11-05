const STORAGE_KEY = "careerlens:user-id";

const generateUserId = () => crypto.randomUUID();

export const getOrCreateUserId = () => {
  const existing = window.localStorage.getItem(STORAGE_KEY);
  if (existing) {
    return existing;
  }
  const created = generateUserId();
  window.localStorage.setItem(STORAGE_KEY, created);
  return created;
};
