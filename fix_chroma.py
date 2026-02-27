"""
Script para limpiar Chroma y re-indexar documentos con los IDs correctos de BD.

Ejecutar desde la raíz del backend:
    python fix_chroma.py

Qué hace:
1. Limpia completamente Chroma (borra todos los chunks)
2. Re-indexa todos los documentos usando los IDs reales de la BD
3. Verifica que la indexación fue exitosa
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("FIX: LIMPIAR Y RE-INDEXAR CHROMA")
print("=" * 60)

# ── 1. Conectar a Chroma ───────────────────────────────────────
print("\n[1] Conectando a Chroma...")
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_DB_PATH = "./storage/Curim_data/chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embeddings
)

collection = vectorstore._collection
total_antes = collection.count()
print(f"   Chunks antes de limpiar: {total_antes}")

# ── 2. Limpiar todos los chunks existentes ─────────────────────
print("\n[2] Limpiando Chroma...")
try:
    all_ids = collection.get()["ids"]
    if all_ids:
        collection.delete(ids=all_ids)
        print(f"   ✅ Eliminados {len(all_ids)} chunks")
    else:
        print("   Chroma ya estaba vacío")
except Exception as e:
    print(f"   ❌ Error limpiando: {e}")
    sys.exit(1)

# ── 3. Cargar documentos desde BD ─────────────────────────────
print("\n[3] Cargando documentos desde la base de datos...")
try:
    from app.db.database import SessionLocal
    from app.models.models import Document

    with SessionLocal() as db:
        documents = db.query(Document)\
                      .filter(
                          Document.text.isnot(None),
                          Document.text != ""
                      )\
                      .all()

    print(f"   Documentos con texto en BD: {len(documents)}")
    for doc in documents:
        print(f"   → ID={doc.id} | {doc.filename} | user={doc.uploaded_by} | {len(doc.text)} chars")

except Exception as e:
    print(f"   ❌ Error cargando documentos: {e}")
    sys.exit(1)

if not documents:
    print("\n⚠️  No hay documentos con texto en la BD.")
    print("   Sube documentos primero desde la interfaz.")
    sys.exit(0)

# ── 4. Re-indexar con IDs correctos ───────────────────────────
print("\n[4] Re-indexando documentos con IDs correctos...")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)

total_chunks = 0
errores = []

for doc in documents:
    try:
        chunks = text_splitter.split_text(doc.text)

        if not chunks:
            print(f"   ⚠️  {doc.filename} no generó chunks")
            continue

        langchain_docs = []
        for i, chunk in enumerate(chunks):
            langchain_docs.append(
                LangchainDocument(
                    page_content=chunk,
                    metadata={
                        "document_id": doc.id,        # ← ID real de la BD
                        "filename": doc.filename,
                        "chunk_index": i,
                        "word_count": len(chunk.split()),
                        "user_id": doc.uploaded_by,   # ← Extra: para filtrar por usuario
                    }
                )
            )

        vectorstore.add_documents(langchain_docs)
        total_chunks += len(chunks)
        print(f"   ✅ {doc.filename} → {len(chunks)} chunks (document_id={doc.id})")

    except Exception as e:
        errores.append((doc.id, doc.filename, str(e)))
        print(f"   ❌ Error indexando {doc.filename}: {e}")

# ── 5. Verificar resultado ─────────────────────────────────────
print("\n[5] Verificando resultado...")
total_despues = collection.count()
print(f"   Chunks antes: {total_antes}")
print(f"   Chunks ahora: {total_despues}")
print(f"   Chunks creados: {total_chunks}")

if errores:
    print(f"\n   ⚠️  Errores ({len(errores)}):")
    for doc_id, fname, err in errores:
        print(f"   ID={doc_id} | {fname} | {err}")

# ── 6. Probar búsqueda con el ID correcto ─────────────────────
print("\n[6] Probando búsqueda con IDs correctos...")
for doc in documents[:2]:
    try:
        results = vectorstore.similarity_search_with_score(
            "información",
            k=3,
            filter={"document_id": {"$in": [doc.id]}}
        )
        print(f"   document_id={doc.id} ({doc.filename}): {len(results)} resultados")
        if results:
            print(f"   Mejor resultado: {results[0][0].page_content[:80]}...")
    except Exception as e:
        print(f"   ❌ Error buscando: {e}")

print("\n" + "=" * 60)
if total_despues > 0 and not errores:
    print("✅ CHROMA RE-INDEXADO CORRECTAMENTE")
    print("   Reinicia el backend y prueba el asistente de voz.")
else:
    print("⚠️  Hay errores — revisa los mensajes anteriores")
print("=" * 60)