> [!QUOTE]
> HOLA


> [!NOTE]
> La interfaz gráfica de usuario desarrollada ofrece una plataforma integral para la visualización interactiva de imágenes médicas DICOM, presentando cortes en los planos axial, sagital y coronal, además de una vista tridimensional del volumen de datos. Una barra de herramientas lateral organiza las funcionalidades de procesamiento de imágenes en cuatro categorías principales: modificaciones de resolución, transformaciones de coordenadas, filtrado y mejoramiento espacial.

> <img width="680" height="917" alt="image" src="https://github.com/user-attachments/assets/2ee04242-8328-4b1c-95f1-4aa6a7384353" />


> [!IMPORTANT]
> Important

> [!TIP]
> Transformaciones de Coordenadas
Este conjunto de herramientas aplica modificaciones geométricas a las imágenes, alterando sus propiedades espaciales.
> •	Rotación: Gira la imagen alrededor de su centro un ángulo especificado por el usuario (en grados).
> •	Traslación: Desplaza la imagen horizontal y verticalmente una cantidad de píxeles definida por el usuario.
> •	Escalamiento: Modifica el tamaño de la imagen aplicando factores de escala independientes para las dimensiones horizontal y vertical.
> •	Inclinación (Shearing): Deforma la imagen aplicando un factor de inclinación en los ejes horizontal y vertical.

> ![transformacion de coordenadas](https://github.com/user-attachments/assets/6fa5a6b2-dd7c-4e1e-9cd4-e66db077065b)

> [!TIP]
> Filtrado
El módulo de filtrado proporciona opciones para procesar la imagen tanto en el dominio espacial como en el frecuencial, con el objetivo de reducir ruido o realzar características.
•	Dominio Frecuencial: 
o	Filtros Pasa Bajas: Atenúan las altas frecuencias para suavizar la imagen y reducir el ruido. Se pueden seleccionar diferentes tipos de ventanas (Gaussiana, Coseno, Bartlett, Hanning, Gaussiana Modificada, Rectangular) y ajustar la dimensión del radio y un factor de ruido.
o	Filtros Pasa Altas: Atenúan las bajas frecuencias para realzar bordes y detalles finos. Utiliza los mismos tipos de ventanas y permite ajustar la dimensión del radio y un factor de borde.
•	Dominio Espacial: 
o	Filtros Pasa Bajas: Aplica un filtro de media con un tamaño de kernel configurable por el usuario para suavizar la imagen. Se puede ajustar un factor de ruido.
o	Filtros Pasa Altas: Implementa varios kernels para la detección de bordes, incluyendo diferentes variaciones de Sobel, Prewitt y Laplace. Se puede ajustar un factor de borde.

![FILTRADO](https://github.com/user-attachments/assets/cbba6edb-a9cd-497c-ac01-9598a67b2c71)

> [!TIP]
> Modificaciones de Resolución
Este módulo ofrece herramientas para alterar las características de resolución de las imágenes, permitiendo observar cómo estos cambios afectan la calidad y la percepción visual.
•	Resolución Espacial: Permite modificar el tamaño de la imagen mediante submuestreo o sobremuestreo. 
o	Submuestreo: Reduce el número de píxeles, lo que puede llevar a una pérdida de detalle. El usuario puede definir un porcentaje de reducción.
o	Sobremuestreo: Aumenta el número de píxeles, por ejemplo, mediante interpolación bilineal, para agrandar la imagen. El usuario puede definir un porcentaje de aumento.
•	Resolución Radiométrica: Ajusta el número de niveles de intensidad (profundidad de bits) de la imagen, lo que afecta el contraste y la cantidad de detalles finos visibles. El usuario puede especificar el número de bits deseado.
•	Resolución Temporal: Simula los efectos del movimiento durante la adquisición de la imagen, introduciendo desenfoque horizontal o vertical. La intensidad del movimiento es configurable por el usuario.

![MODIFICACION DE RESOLUCION](https://github.com/user-attachments/assets/36152dea-86c3-4a60-a814-d17ac787a595)


> [!TIP]
> Mejoramiento Espacial
Este módulo está diseñado para mejorar la calidad de las imágenes que puedan estar degradadas, utilizando algoritmos de restauración.
•	Permite aplicar métodos como Constrained Least Squares (CLS), Weighted Constrained Least Squares (WCLS) y Bayesian Mean Restoration (BMR).
•	El usuario puede definir el ancho de la función de dispersión de puntos (PSF).
•	Se muestran métricas cuantitativas para evaluar el desempeño de cada algoritmo, tales como: 
o	Peak Signal-to-Noise Ratio (PSNR)
o	Increment in Signal-to-Noise Ratio (IOSNR)
o	Mean Absolute Error (MAE)
o	Structural Similarity Index (SSIM)
•	Los resultados de los diferentes métodos de mejoramiento, la imagen degradada y la imagen original pueden alternarse en las vistas para una comparación directa.

![MEJORAMIENTO ESPACIAL](https://github.com/user-attachments/assets/4a3b8ca5-2733-4f1c-b507-9e47bc0e008f)

> [!WARNING]
> Warning

> [!CAUTION]
> Caution

> > [!HOLA]
