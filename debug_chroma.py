"""
Script de diagnóstico para verificar el estado del vectorstore Chroma.
Ejecutar desde la raíz del backend:
    python debug_chroma.py

Muestra:
- Cuántos chunks hay en Chroma
- Qué metadata tienen los chunks
- Si los document_id coinciden con los de la BD
- Si el filtro funciona correctamente
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("DIAGNÓSTICO DE CHROMA VECTORSTORE")
print("=" * 60)

# ── 1. Conectar a Chroma directamente ──────────────────────────
print("\n[1] Conectando a Chroma...")
try:
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    VECTOR_DB_PATH = "./storage/Curim_data/chroma_db"

    if not os.path.exists(VECTOR_DB_PATH):
        print(f"❌ La carpeta {VECTOR_DB_PATH} NO EXISTE")
        print("   → Los documentos nunca fueron indexados en Chroma")
        sys.exit(1)

    files = os.listdir(VECTOR_DB_PATH)
    print(f"✅ Carpeta existe. Archivos: {files}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )
    print("✅ Chroma conectado")
except Exception as e:
    print(f"❌ Error conectando a Chroma: {e}")
    sys.exit(1)

# ── 2. Contar total de chunks ───────────────────────────────────
print("\n[2] Contando chunks en Chroma...")
try:
    collection = vectorstore._collection
    total = collection.count()
    print(f"   Total de chunks en Chroma: {total}")

    if total == 0:
        print("❌ CHROMA ESTÁ VACÍO — los documentos no se indexaron")
        print("   Solución: re-sube los documentos para que se indexen")
        sys.exit(1)
except Exception as e:
    print(f"❌ Error contando: {e}")

# ── 3. Ver metadata de los primeros 5 chunks ───────────────────
print("\n[3] Metadata de los primeros 5 chunks...")
try:
    sample = collection.get(limit=5, include=["metadatas", "documents"])
    for i, (meta, doc) in enumerate(zip(sample["metadatas"], sample["documents"])):
        print(f"\n   Chunk {i+1}:")
        print(f"   Metadata: {meta}")
        print(f"   Texto (50 chars): {doc[:50]}...")
except Exception as e:
    print(f"❌ Error obteniendo sample: {e}")

# ── 4. Ver qué document_ids existen en Chroma ──────────────────
print("\n[4] Verificando campo 'document_id' en metadata...")
try:
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    doc_ids_en_chroma = set()
    sin_document_id = 0

    for meta in all_meta:
        if "document_id" in meta:
            doc_ids_en_chroma.add(meta["document_id"])
        else:
            sin_document_id += 1

    print(f"   document_ids en Chroma: {sorted(doc_ids_en_chroma)}")
    print(f"   Chunks SIN document_id: {sin_document_id}")

    if sin_document_id > 0:
        print(f"⚠️  HAY {sin_document_id} CHUNKS SIN document_id")
        print("   → El filtro por document_id no funcionará para estos chunks")
        print("   → El RAG retornará 'No se encontró información relevante'")
except Exception as e:
    print(f"❌ Error: {e}")

# ── 5. Comparar con IDs de la base de datos ────────────────────
print("\n[5] Comparando con documentos en la base de datos...")
try:
    from app.db.database import SessionLocal
    from app.models.models import Document

    with SessionLocal() as db:
        docs_en_bd = db.query(Document.id, Document.filename, Document.uploaded_by)\
                       .filter(Document.text.isnot(None))\
                       .all()

    print(f"   Documentos en BD con texto: {len(docs_en_bd)}")
    for doc in docs_en_bd[:10]:
        en_chroma = doc.id in doc_ids_en_chroma
        estado = "✅ en Chroma" if en_chroma else "❌ NO está en Chroma"
        print(f"   ID={doc.id} | {doc.filename} | user={doc.uploaded_by} | {estado}")

    ids_bd = {d.id for d in docs_en_bd}
    ids_faltantes = ids_bd - doc_ids_en_chroma
    if ids_faltantes:
        print(f"\n   ⚠️  IDs en BD pero NO en Chroma: {ids_faltantes}")
        print("   → Estos documentos no son buscables por voz ni chat")

except Exception as e:
    print(f"❌ Error comparando con BD: {e}")

# ── 6. Probar búsqueda con filtro ──────────────────────────────
print("\n[6] Probando búsqueda SIN filtro...")
try:
    results = vectorstore.similarity_search_with_score("información del documento", k=3)
    print(f"   Resultados sin filtro: {len(results)}")
    for doc, score in results:
        print(f"   score={score:.3f} | meta={doc.metadata} | texto={doc.page_content[:60]}...")
except Exception as e:
    print(f"❌ Error en búsqueda sin filtro: {e}")

if doc_ids_en_chroma:
    print(f"\n[7] Probando búsqueda CON filtro por document_id={list(doc_ids_en_chroma)[0]}...")
    try:
        test_id = list(doc_ids_en_chroma)[0]
        results = vectorstore.similarity_search_with_score(
            "información",
            k=3,
            filter={"document_id": {"$in": [test_id]}}
        )
        print(f"   Resultados con filtro: {len(results)}")
        for doc, score in results:
            print(f"   score={score:.3f} | {doc.page_content[:60]}...")
    except Exception as e:
        print(f"❌ Error en búsqueda con filtro: {e}")
        print("   → El problema puede ser la sintaxis del filtro de Chroma")

print("\n" + "=" * 60)
print("FIN DEL DIAGNÓSTICO")
print("=" * 60)
