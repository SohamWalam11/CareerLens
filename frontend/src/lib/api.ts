import axios from "axios";

const apiClient = axios.create({
  baseURL: __API_BASE_URL__
});

export default apiClient;
