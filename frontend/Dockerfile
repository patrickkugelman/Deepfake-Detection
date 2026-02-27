FROM node:18-alpine
WORKDIR /app
COPY package.json .
RUN npm install --legacy-peer-deps
COPY . .
EXPOSE 8501
# Pornim în modul dev pentru a vedea modificările live și a evita build complex acum
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "8501"]