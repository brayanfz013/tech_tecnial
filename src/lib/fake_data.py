import random

import pandas as pd  # type: ignore
from faker import Faker


class FakeDataPSQL:
    def __init__(self) -> None:
        # Crear una instancia de self.Faker
        self.fake = Faker()

    # Función para generar datos falsos para la tabla customers
    def generate_customers(self, num_customers):
        customers = []
        for _ in range(num_customers):
            customer_id = self.fake.random_number(digits=6, fix_len=True)
            email = self.fake.unique.email()
            first_name = self.fake.first_name()
            last_name = self.fake.last_name()
            gender = random.choice(["M", "F", "O"])
            date_of_birth = self.fake.date_of_birth(minimum_age=18, maximum_age=80)
            phone = self.fake.phone_number()
            address = self.fake.address()
            is_seller = self.fake.boolean(chance_of_getting_true=50)
            created_at = self.fake.date_time_this_decade()
            updated_at = created_at
            customers.append(
                (
                    customer_id,
                    email,
                    first_name,
                    last_name,
                    gender,
                    date_of_birth,
                    phone,
                    address,
                    is_seller,
                    created_at,
                    updated_at,
                )
            )
        return customers

    # Función para generar categorías
    def generate_categories(self, num_categories):
        categories = []
        for i in range(1, num_categories + 1):
            category_id = i
            name = self.fake.unique.word()
            parent_id = random.choice([None] + list(range(1, i)))
            level = 0 if parent_id is None else 1
            path = (
                f"/{category_id}"
                if parent_id is None
                else f"/{parent_id}/{category_id}"
            )
            created_at = self.fake.date_time_this_decade()
            updated_at = created_at
            categories.append(
                (category_id, name, parent_id, level, path, created_at, updated_at)
            )
        return categories

    # Función para generar items
    def generate_items(
        self, num_items, table_customer: pd.DataFrame, table_ctg: pd.DataFrame
    ):
        items = []
        for _ in range(num_items):
            item_id = self.fake.random_number(digits=6, fix_len=True)
            seller_id = int(
                table_customer.iloc[random.randint(1, 99)]["customer_id"]
            )  # Suponiendo que ya se han generado 100 usuarios
            category_id = int(
                table_ctg.iloc[random.randint(0, 19)]["category_id"]
            )  # Suponiendo que ya se han generado 20 categorías
            name = self.fake.word()
            description = self.fake.text(max_nb_chars=200)
            price = round(random.uniform(5.0, 500.0), 2)
            # status = random.choice(["active", "inactive", "deleted"])
            status = random.choice(range(1, 4))
            deletion_date = self.fake.date_time_this_decade() if status == 3 else None
            created_at = self.fake.date_time_this_decade()
            updated_at = created_at
            items.append(
                (
                    item_id,
                    seller_id,
                    category_id,
                    name,
                    description,
                    price,
                    status,
                    deletion_date,
                    created_at,
                    updated_at,
                )
            )
        return items

    # Función para generar órdenes
    def generate_orders(
        self, num_orders: int, table_customer: pd.DataFrame, table_item: pd.DataFrame
    ):
        orders = []
        for _ in range(num_orders):
            ord_id = self.fake.random_number(digits=6, fix_len=True)
            buyer_id = int(table_customer.iloc[random.randint(1, 99)]["customer_id"])
            item_id = int(table_item.iloc[random.randint(1, 48)]["item_id"])
            quantity = random.randint(1, 10)
            total_price = round(quantity * random.uniform(5.0, 500.0), 2)
            ord_date = self.fake.date_time_this_decade()
            status_ord = random.randint(1, 4)
            created_at = ord_date
            updated_at = created_at

            orders.append(
                (
                    ord_id,
                    buyer_id,
                    item_id,
                    quantity,
                    total_price,
                    ord_date,
                    status_ord,
                    created_at,
                    updated_at,
                )
            )
        return orders
