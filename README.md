# Prueba técnica 
> **Consideraciones**: 
Entendemos cada situación en particular y por eso vamos a dejar que hagas la prueba a tu tiempo y
con las herramientas que consideres. En el correo encuentras la fecha de entrega.

**Puntos a evaluar:**

-  Orden y comentarios en los entregables (que pueden ser código, scripts, imágenes,
diagramas etc)
-  Entregables de tipo queries o código de la manera más optima posible.
-  Entendimiento del problema.
-  Tiempo de entrega

## Primera parte – Sql
### **Objetivo**
Responder preguntas sobre un modelo de base de datos, diseñado a partir de una necesidad de
negocio.

**Descripción de la necesidad**

Teniendo en cuenta el modelo de ecommerce que maneja alguna compañía, tenemos algunas
entidades básicas que queremos representar: Customer, Order, Item y Category.

- *`Customer`*: Es la entidad donde se encuentran todos nuestros usuarios, ya sean Buyers o
Sellers del Site. Los principales atributos son email, nombre, apellido, sexo, dirección, fecha
de nacimiento, teléfono, entre otros.

- *`Item`*: Es la entidad donde se encuentran los productos publicados en el marketplace. El
volumen es muy grande debido a que se encuentran todos los productos que en algún
momento fueron publicados. Mediante el estado del ítem o fecha de baja se puede detectar
los ítems activos del marketplace.

- *`Category`*: Es la entidad donde se encuentra la descripción de cada categoría con su
respectivo path. Cada ítem tiene asociado una categoría.

- *`Order`*: La order es la entidad que refleja las transacciones generadas dentro del site (cada
compra es una order). En este caso no vamos a contar con un flujo de carrito de compras
por lo tanto cada ítem que se venda será reflejado en una order independientemente de la
cantidad que se haya comprado.

Un usuario ingresa al sitio para comprar dos dispositivos móviles iguales. Realiza la búsqueda
navegando por las categorías Tecnología > Celulares y Teléfonos > Celulares y Smartphones, y
finalmente encuentra el producto que necesita comprar. Procede con la compra del mismo
seleccionado dos unidades, el cual genera una orden de compra.

### ¿Que debes enviar de este punto?
1. Un modelo entidad relación (de manera grafica) que logre responder las preguntas que vas
a responder en el punto 3.
2. Generar los scripts DDL para la creación de las tablas. Debe estar en un documento .sql
nombrado como “creacion_tablas”
3. En lenguaje sql debes responder las siguientes consultas, deben quedar en un documento
.sql nombrado como “solución_queries”:

    a. Listar los usuarios que cumplan años el día de hoy cuya cantidad de ventas
realizadas en enero 2020 sea superior a 1500.

    b. Por cada mes del 2020, se solicita el top 5 de usuarios que más vendieron($) en la
categoría Celulares. Se requiere el mes y año de análisis, nombre y apellido del
vendedor, cantidad de ventas realizadas, cantidad de productos vendidos y el
monto total transaccionado.

    c. Se solicita poblar una nueva tabla con el precio y estado de los Ítems a fin del día.
Tener en cuenta que debe ser reprocesable. Vale resaltar que en la tabla Item,
vamos a tener únicamente el último estado informado por la PK definida. (Se puede
resolver a través de StoredProcedure).

## Segunda parte – ML
### **Objetivo**
Verificar el conocimiento que cuenta de Machine Learning

**Descripción de la necesidad**

En la plataforma de datos abiertos de Medellín ( http://medata.gov.co/ ) puede encontrar mas de
500 conjuntos de datos con los cuales puede trabajar de diferentes secretarías. La idea es que con
uno que usted escoja entrene un modelo de ML a consideración.

**¿Qué debes enviar en este punto?**
1. Un documento .pdf explicando los motivos por los cuales escogiste el conjunto de datos con
el que vas a solucionar la prueba, aparte de una explicación del modelo que vas a usar y el
porqué, además de los que esperas predecir con el modelo.
2. Un documento en el lenguaje de programación a escoger con la solución.

## Tercera parte – Api’s
### **Objetivo**
Realizar un análisis sobre la oferta/vidriera de las opciones de productos que responden a distintas
búsquedas en el sitio Mercadolibre.com.ar
Importante: este punto es opcional

**Descripción de la necesidad**

-  Barrer una lista de más de 150 ítems ids en el servicio público:
https://api.mercadolibre.com/sites/MLA/search?q=chromecast&limit=50#json En este
caso particular y solo a modo de ejemplo, son resultados para la búsqueda “chromecast”,
pero deberás elegir otros términos para el experimento que permitan enriquecer el análisis
en un hipotético dashboard (ejemplo Google Home, Apple TV, Amazon Fire TV, o afines para
poder comparar dispositivos portátiles, o bien elegir otros 3 que te interesen para
comparar).

-  Por cada resultado, realizar el correspondiente GET por Item_Id al recurso público:
https://api.mercadolibre.com/items/{Item_Id}

-  Escribir los resultados en un archivo plano delimitado por comas, desnormalizando el JSON
obtenido en el paso anterior, en tantos campos como sea necesario para guardar las
variables que te interesen modelar.

**¿Qué debes enviar en este punto?**

    Un archivo en el lenguaje de programación preferido con la solución.



# Solucione y entregables
Se entrega el repositorio completo con los enlaces a los archivos igualemente se envia un zip con los documentos aparter pedido
los para ejecutar el codigo se tiene las especificaciones del versionamiento de python como de los paquetes en el archivo [`pyproject`](pyproject.toml)

## Primera parte – Sql
los archivos de la socion estan son estos enlaces [`“creacion_tablas”`](src/data/sql/creacion_tablas.sql) y [`“solución_queries”`](src/data/sql/solución_queries.sql).


## Segunda parte – ML
EL PDF explicando el codigo se encuetra [`aqui`](src/SegundoPunto.pdf) igualmente esta su version en [`html`](src/SegundoPunto.html) para cualquiera tipo de revision 

El codigo solucion se encuentra tanto en [`notebook`](src/notebooks/PrimerPunto_educacion.ipynb) como en [`script`](<src/Segunda Parte ML.py>)


## Tercera parte – Api’s
El archivo de python generado se encuentra en : [`Codigo solucion`](<src/Tercera Parte API.py>)