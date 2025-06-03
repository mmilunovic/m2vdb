
# 🧠 m2vdb Frontend

> A chaotic little playground for exploring your vector data in glorious 3D.

This is the **interactive web UI** for [m2vdb](https://github.com/your_username/m2vdb) — a tiny but mighty vector database built for learning, hacking, and yelling “WHY IS THIS VECTOR OVER THERE?!” in your browser.


## 🚀 What is this?

This app lets you:

- 🎯 Visualize your vector space in 3D
- 🕸️ See connections between vectors (k-NN as edges)
- 📄 Add new documents and query them live
- 🤔 Build intuition about vector search & embeddings

It’s like a microscope for your embeddings — but cooler looking.


## 🛠️ Tech Stack (aka the usual suspects)

- ⚛️ **React** – component-based UI wizardry
- ⚡ **Vite** – fast bundling because we don’t have time to wait
- 🔠 **TypeScript** – some light type therapy
- 🧱 **Ant Design** – nice buttons, minimal CSS crying
- 🌐 **vis-network** – drawing nodes so you don’t have to
- 📡 **Axios** – talking to your backend with minimal fuss


## 🧪 Getting Started

1. **Install dependencies:**
  ```bash
    npm install
  ```

2. **Run it locally:**

  ```bash
  npm run dev
  ```

   Open [http://localhost:5173](http://localhost:5173) — behold your vectors in full browser glory.

3. **Backend setup:**
   Make sure your FastAPI backend is running and exposing routes like:

   * `POST /add`
   * `POST /search`
   * `GET /vectors`
     …or whatever you cooked up in your API.


## 🧩 Features (some done, some delusional)

* ✅ 3D vector graph (with PCA magic)
* ✅ Clickable nodes (view text, metadata, connections)
* ✅ Add/query from the browser
* 🚧 Highlight search results (soon!)
* 🚧 Upload full documents
* 🚧 Dark mode, because yes


## 💡 Why?

Because vectors are cool, and you deserve better than a CSV file and despair.


## 🧠 Inspired By

* [LightRAG WebUI](https://github.com/HKUDS/LightRAG/tree/main/lightrag_webui) — they walked so we could run with scissors
* Every time you’ve stared at a vector and thought, “...but what does it *mean*?”


## 🤝 Contributions?
Absolutely. Fork, break it, PR it. This is a learning project, not a bank app.

## 📚 See Also

* [🧬 m2vdb backend (Python)](../README.md)
* [🔍 The original m2vdb post](https://www.linkedin.com/posts/...)