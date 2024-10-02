-- Creacion de tabla de usuarios
CREATE TABLE IF NOT EXISTS customers (
  customer_id BIGINT PRIMARY KEY,
  email VARCHAR(255) UNIQUE NOT NULL,
  first_name VARCHAR(100) NOT NULL,
  last_name VARCHAR(100) NOT NULL,
  gender CHAR(1) NOT NULL CHECK (gender IN ('M', 'F', 'O')),
  date_of_birth DATE NOT NULL,
  phone VARCHAR(20),
  address TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  is_seller BOOLEAN DEFAULT FALSE
);

-- CREATE INDEX idx_email ON customers (email);
-- CREATE INDEX idx_date_of_birth ON customers (date_of_birth);

-- Creacion de la tabla de categorias
CREATE TABLE IF NOT EXISTS category (
    category_id BIGINT PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    parent_id BIGINT,
    level INT NOT NULL DEFAULT 0,
    path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_parent_id FOREIGN KEY (parent_id) REFERENCES category (category_id)
);

-- CREATE INDEX idx_parent_id ON category (parent_id);
-- CREATE INDEX idx_name ON category (name);

-- Creacion de la tabla de statusitem
CREATE TABLE IF NOT exists statusitem(
  status_item_id INT PRIMARY KEY,
  status_item  VARCHAR(50) NOT NULL CHECK (status_item IN ('active', 'inactive', 'deleted'))
);

-- Creacion de la tabla de items
CREATE TABLE IF NOT exists item(
  item_id BIGINT PRIMARY KEY,
  seller_id BIGINT NOT NULL,
  category_id BIGINT NOT NULL,
  name VARCHAR(255) NOT NULL,
  description TEXT,
  price DECIMAL(10,2) NOT NULL,
  status_item INT NOT NULL,
  deletion_date TIMESTAMP,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  CONSTRAINT fk_status_id FOREIGN KEY (status_item) REFERENCES statusitem (status_item_id),
  CONSTRAINT fk_seller_id FOREIGN KEY (seller_id) REFERENCES customers (customer_id),
  CONSTRAINT fk_category_id FOREIGN KEY (category_id) REFERENCES category (category_id)

);

-- CREATE INDEX idx_seller_id ON item (seller_id);
-- CREATE INDEX idx_category_id ON item (category_id);
-- CREATE INDEX idx_status_item ON item (status_item);


-- Creacion de la tabla de status_order
CREATE TABLE IF NOT exists statusord(
  status_ord_id INT PRIMARY KEY,
  status_ord   VARCHAR(50) NOT NULL CHECK (status_ord IN ('pending', 'completed', 'shipped', 'cancelled'))
);


-- Creacion de la tabla de ordenes
CREATE TABLE IF NOT exists ordes(
  ord_id BIGINT PRIMARY KEY,
  buyer_id BIGINT NOT NULL,
  item_id BIGINT NOT NULL,
  quantity INT NOT NULL CHECK (quantity > 0),
  total_price DECIMAL(10,2) NOT NULL,
  ord_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  status_ord INT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  CONSTRAINT fk_status_id FOREIGN KEY (status_ord) REFERENCES statusord (status_ord_id),
  CONSTRAINT fk_buyer_id FOREIGN KEY (buyer_id) REFERENCES customers (customer_id),
  CONSTRAINT fk_item_id FOREIGN KEY (item_id) REFERENCES item (item_id)

);

-- CREATE INDEX idx_buyer_id ON orders (buyer_id);
-- CREATE INDEX idx_item_id ON orders (item_id);
-- CREATE INDEX idx_ord_date ON orders (ord_date);
