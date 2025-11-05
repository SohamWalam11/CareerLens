# Build stage
FROM node:20-alpine as builder

WORKDIR /build

COPY package*.json ./
RUN npm ci --prefer-offline

COPY . .
RUN npm run build

# Runtime stage - nginx serving static SPA
FROM nginx:alpine

COPY --from=builder /build/dist /usr/share/nginx/html

# Nginx config for SPA (all routes -> index.html)
RUN echo 'server {\n\
  listen 5173;\n\
  root /usr/share/nginx/html;\n\
  index index.html;\n\
\n\
  location / {\n\
    try_files $uri $uri/ /index.html;\n\
  }\n\
\n\
  location ~* \\.(?:css|js|svg|png|jpg|jpeg|gif|ico)$ {\n\
    expires 1y;\n\
    add_header Cache-Control "public, immutable";\n\
  }\n\
}' > /etc/nginx/conf.d/default.conf

EXPOSE 5173

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:5173/index.html || exit 1

CMD ["nginx", "-g", "daemon off;"]
