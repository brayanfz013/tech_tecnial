import asyncio
import csv

import httpx

# Definir los términos de búsqueda que deseas utilizar
search_terms = ["Google Home", "Apple TV", "Amazon Fire TV"]
base_search_url = "https://api.mercadolibre.com/sites/MLA/search"
base_item_url = "https://api.mercadolibre.com/items/"

# Definir el archivo CSV de salida
output_file = "items_data_async.csv"


# Función asincrónica para obtener los ítems a partir de un término de búsqueda
async def fetch_items_by_search_term(client, term, limit=50):
    response = await client.get(base_search_url, params={"q": term, "limit": limit})
    response.raise_for_status()  # Asegura que no falle la request
    data = response.json()
    return data["results"]


# Función asincrónica para obtener los detalles de un ítem específico
async def fetch_item_details(client, item_id):
    response = await client.get(f"{base_item_url}{item_id}")
    response.raise_for_status()
    return response.json()


# Función para guardar los datos en CSV
def save_items_to_csv(items, output_file):
    fieldnames = [
        "id",
        "title",
        "category_id",
        "price",
        "currency_id",
        "available_quantity",
        "sold_quantity",
        "condition",
        "permalink",
        "seller_id",
        "seller_power_seller_status",
    ]

    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for item in items:
            seller_info = item.get("seller", {})
            row = {
                "id": item.get("id"),
                "title": item.get("title"),
                "category_id": item.get("category_id"),
                "price": item.get("price"),
                "currency_id": item.get("currency_id"),
                "available_quantity": item.get("available_quantity"),
                "sold_quantity": item.get("sold_quantity"),
                "condition": item.get("condition"),
                "permalink": item.get("permalink"),
                "seller_id": seller_info.get("id"),
                "seller_power_seller_status": seller_info.get("power_seller_status"),
            }
            writer.writerow(row)


# Función principal asincrónica para manejar la concurrencia
async def main():
    all_items = []

    async with httpx.AsyncClient() as client:
        # Obtener los items para cada término de búsqueda en paralelo
        search_tasks = [
            fetch_items_by_search_term(client, term) for term in search_terms
        ]

        search_results = await asyncio.gather(*search_tasks)

        # Obtener detalles de cada ítem por su ID en paralelo
        item_ids = [item["id"] for items in search_results for item in items]
        detail_tasks = [fetch_item_details(client, item_id) for item_id in item_ids]
        all_items = await asyncio.gather(*detail_tasks)

    # Guardar los ítems obtenidos en un archivo CSV
    save_items_to_csv(all_items, output_file)
    print(f"Datos guardados en {output_file}")


# Ejecutar el bucle de eventos
if __name__ == "__main__":
    asyncio.run(main())
