-- a. Listar los usuarios que cumplan años el día de hoy cuya cantidad de ventas realizadas en enero 2020 
-- sea superior a 1500.
SELECT c.customer_id, c.first_name 
FROM customers c
JOIN (
  SELECT buyer_id, SUM(total_price) AS total_ventas
  FROM ordes
  WHERE EXTRACT(YEAR FROM ord_date) = 2020
  AND EXTRACT(MONTH FROM ord_date) = 1
  GROUP BY buyer_id
  HAVING SUM(total_price) > 1500
) o ON c.customer_id = o.buyer_id
WHERE EXTRACT(MONTH FROM c.date_of_birth) = EXTRACT(MONTH FROM CURRENT_DATE)
AND EXTRACT(DAY FROM c.date_of_birth) = EXTRACT(DAY FROM CURRENT_DATE);

-- b. Por cada mes del 2020, se solicita el top 5 de usuarios que más vendieron($) en la categoría Celulares. 
-- Se requiere el mes y año de análisis, nombre y apellido del vendedor, cantidad de ventas realizadas, cantidad de productos vendidos y el monto total transaccionado.
WITH ventas_celulares AS (
  SELECT 
    EXTRACT(YEAR FROM o.ord_date) AS año,
    EXTRACT(MONTH FROM o.ord_date) AS mes,
    c.first_name AS nombre_vendedor,
    c.last_name AS apellido_vendedor,
    COUNT(o.ord_id) AS cantidad_ventas,
    SUM(o.quantity) AS cantidad_productos_vendidos,
    SUM(o.total_price) AS monto_total_transaccionado
  FROM ordes o
  JOIN item i ON o.item_id = i.item_id
  JOIN customers c ON o.buyer_id = c.customer_id
  JOIN category cat ON i.category_id = cat.category_id
  WHERE cat.name = 'Celulares'
  AND o.ord_date >= '2020-01-01'
  AND o.ord_date < '2021-01-01'
  GROUP BY 
    EXTRACT(YEAR FROM o.ord_date),
    EXTRACT(MONTH FROM o.ord_date),
    c.first_name,
    c.last_name
)
SELECT 
  año,
  mes,
  nombre_vendedor,
  apellido_vendedor,
  cantidad_ventas,
  cantidad_productos_vendidos,
  monto_total_transaccionado
FROM (
  SELECT 
    *,
    ROW_NUMBER() OVER (PARTITION BY año, mes ORDER BY monto_total_transaccionado DESC) AS row_num
  FROM ventas_celulares
) sub
WHERE row_num <= 5
ORDER BY año, mes, monto_total_transaccionado DESC;


-- c. Se solicita poblar una nueva tabla con el precio y estado de los Ítems a fin del día. 
-- Tener en cuenta que debe ser reprocesable. Vale resaltar que en la tabla Item,
-- vamos a tener únicamente el último estado informado por la PK definida. 
-- (Se puede resolver a través de StoredProcedure).
CREATE TABLE IF NOT EXISTS item_historico (
  item_historico_id SERIAL PRIMARY KEY,
  item_id BIGINT NOT NULL,
  fecha DATE NOT NULL,
  price DECIMAL(10,2) NOT NULL,
  status VARCHAR(50) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_fecha ON item_historico (fecha);

CREATE OR REPLACE PROCEDURE poblar_item_diario()
LANGUAGE plpgsql
AS $$
BEGIN
    -- Eliminar registros del día actual para asegurar la reprocesabilidad
    DELETE FROM item_historico 
    WHERE fecha = CURRENT_DATE;

    -- Insertar los datos del día actual
    INSERT INTO item_historico (item_id, price, status, fecha)
    SELECT i.item_id, i.price, i.status_item, CURRENT_DATE
    FROM item i;
END;
$$;

CALL poblar_item_diario();
