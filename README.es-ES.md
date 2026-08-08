
</think>

# RLPF: Aprendizaje por Refuerzo con Retroalimentación Física

Un framework flexible para implementar aprendizaje por refuerzo con retroalimentación física utilizando modelos de difusión, construido sobre la base del framework `verl_diffusion`.

## Descripción General

El framework RLPF (Reinforcement Learning with Physical Feedback / Aprendizaje por Refuerzo con Retroalimentación Física) proporciona un enfoque estructurado para implementar modelos de difusión con capacidades de aprendizaje por refuerzo y mecanismos de retroalimentación física. Está diseñado para ser modular y extensible, permitiéndote integrar tus propios modelos, funciones de recompensa y procedimientos de entrenamiento, aprovechando al mismo tiempo la sólida base de `verl_diffusion`.

## Estructura del Framework

El framework RLPF se organiza en varios componentes clave, construidos sobre la arquitectura de `verl_diffusion`:

- **Model**: Define la arquitectura del modelo de difusión
- **Trainer**: Maneja el proceso de entrenamiento
- **Worker**: Contiene componentes para rollouts, recompensas y filtrado
- **Utils**: Funciones auxiliares y utilidades
- **Dataloader**: Carga de datos y preprocesamiento

## 📦 Instalación

### 1. Configurar el Entorno de Python

```bash
conda create -n RLPF python=3.10.14
conda activate RLPF
pip install -r requirements.txt
```

### 2. Instalar xtb para Cálculos de Fuerzas (Usado en la Recompensa)

```bash
cd RLPF
git clone https://github.com/grimme-lab/xtb-python.git
cd xtb-python
conda install mkl mkl-devel
conda install -c conda-forge "gfortran<12"
conda install -c conda-forge mkl mkl-devel blas lapack
pip install .
cd ..
```

## 🚀 Cómo usar RLPF para fine-tunear EDM con recompensa xtb 
### 1. Configurar el entorno de EDM
```bash
cd ./Model/EDM
pip install .
```
### 2. Para la generación de moléculas QM9
```bash
bash example/edm_ddpo_xtb/run.sh
```

### 3. Evaluación en la generación de moléculas QM9
```bash
bash example/edm_ddpo_xtb/run_eval.sh
```

## Primeros Pasos con RLPF

El framework RLPF te permite ajustar finamente modelos de difusión como EDM utilizando recompensas de retroalimentación física, como los cálculos de XTB (Extended Tight Binding / Acoplamiento Fuerte Extendido). A continuación, se muestra cómo comenzar:

### 1. Define tu Modelo Personalizado

Para usar tu propio modelo con el framework RLPF, necesitas crear una clase que herede de `BaseModel` en el paquete `verl_diffusion`:

```python
from verl_diffusion.model.base import BaseModel

class YourCustomModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        # Initialize your model components
        
    def sample(self, n_samples, n_nodes, node_mask, edge_mask, context=None, fix_noise=False, timestep=1000):
        """
        Draw samples from your generative model.
        
        Args:
            n_samples: Number of samples to generate
            n_nodes: Number of nodes per sample
            node_mask: Mask for nodes
            edge_mask: Mask for edges
            context: Optional context information
            fix_noise: Whether to use fixed noise
            timestep: Number of diffusion timesteps
            
        Returns:
            Generated samples and associated information
        """
        # Implement your sampling logic
        pass
        
    def compute_log_p_zs_given_zt(self, x, mu, sigma, node_mask=None):
        """
        Compute log probability of zs given zt.
        
        Args:
            x: Input tensor
            mu: Mean tensor
            sigma: Standard deviation tensor
            node_mask: Optional node mask
            
        Returns:
            Log probability values
        """
        # Implement your probability computation
        pass
        
    def sample_p_zs_given_zt(self, s, t, zt, node_mask, edge_mask, context=None, fix_noise=False):
        """
        Sample from p(zs | zt).
        
        Args:
            s: Source timestep
            t: Target timestep
            zt: Latent at timestep t
            node_mask: Node mask
            edge_mask: Edge mask
            context: Optional context
            fix_noise: Whether to use fixed noise
            
        Returns:
            Sampled values and associated information
        """
        # Implement your conditional sampling
        pass
```

### 2. Define tu Función de Recompensa de Retroalimentación Física

Crea una función de recompensa que herede de la clase base de recompensas:

```python
from verl_diffusion.worker.reward.base import BaseReward

class YourCustomReward(BaseReward):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your reward components
        
    def calculate_rewards(self, samples):
        """
        Calculate rewards for generated samples.
        
        Args:
            samples: Generated samples
            
        Returns:
            Reward values
        """
        # Implement your reward calculation
        pass
```

### 3. Configura tu Proceso de Entrenamiento RLPF

Crea un archivo de configuración que especifique los parámetros de entrenamiento RLPF y la configuración de retroalimentación física:

```python
config = {
    "model": {
        "diffusion_steps": 1000,
        "diffusion_noise_schedule": "cosine",
        "diffusion_noise_precision": 1e-5,
        "diffusion_loss_type": "l2",
        "normalize_factors": [1.0, 1.0, 1.0],
        "include_charges": True
    },
    "train": {
        "batch_size": 32,
        "micro_batch_size": 8,
        "learning_rate": 1e-4,
        "clip_advantage_value": 5.0,
        "save_path": "./exp/your_model"
    },
    "dataloader": {
        "epoches": 100
    },
    "wandb": {
        "enabled": True,
        "project": "your-project",
        "name": "your-run-name"
    }
}
```

### 4. Configura la Pipeline de Entrenamiento RLPF

```python
from verl_diffusion.trainer.ddpo_trainer import DDPOTrainer
from verl_diffusion.dataloader.dataloader import EDMDataLoader

# Initialize your components
model = YourCustomModel(config)
dataloader = EDMDataLoader(config)
rollout = YourCustomRollout(config, model)
rewarder = YourCustomReward(config)
actor = YourCustomActor(config)

# Create the trainer
trainer = DDPOTrainer(
    config=config,
    model=model,
    dataset_info=dataset_info,
    device=device,
    dataloader=dataloader,
    rollout=rollout,
    rewarder=rewarder,
    actor=actor
)

# Start training
trainer.fit()
```

## Componentes Clave de RLPF

El framework RLPF aprovecha los siguientes componentes clave de `verl_diffusion`, añadiendo capacidades de retroalimentación física:

### DataProto

El framework utiliza `DataProto` para el intercambio de datos entre componentes. Proporciona una forma estandarizada de manejar datos tensoriales y no tensoriales:

```python
from verl_diffusion.protocol import DataProto

# Create a DataProto
data = DataProto.from_dict(
    tensors={"key1": tensor1, "key2": tensor2},
    non_tensors={"key3": array1},
    meta_info={"info": "metadata"}
)

# Access data
tensor_data = data.batch["key1"]
non_tensor_data = data.non_tensor_batch["key3"]
meta_data = data.meta_info["info"]
```

### Rollout

El componente Rollout maneja el proceso de generación:

```python
from verl_diffusion.worker.rollout.base import BaseRollout

class YourCustomRollout(BaseRollout):
    def generate_minibatch(self, batch):
        """
        Generate samples for a mini-batch.
        
        Args:
            batch: Input batch data
            
        Returns:
            Generated samples
        """
        # Implement your generation logic
        pass
```

### Actor

El componente Actor define la política para la selección de acciones:

```python
from verl_diffusion.worker.actor.base import BaseActor

class YourCustomActor(BaseActor):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your actor components
        
    def act(self, state):
        """
        Select actions based on the current state.
        
        Args:
            state: Current state
            
        Returns:
            Selected actions
        """
        # Implement your action selection logic
        pass
```

## Uso Avanzado

### Filtros Personalizados

Puedes implementar filtros personalizados para procesar las muestras generadas:

```python
from verl_diffusion.worker.filter.base import BaseFilter

class YourCustomFilter(BaseFilter):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your filter components
        
    def filter(self, samples):
        """
        Filter generated samples.
        
        Args:
            samples: Generated samples
            
        Returns:
            Filtered samples
        """
        # Implement your filtering logic
        pass
```

### Procesamiento Paralelo

El framework soporta el procesamiento paralelo utilizando Ray:

```python
import ray

# Initialize Ray
ray.init()

# Use Ray for parallel processing in your components
@ray.remote
def parallel_process(data):
    # Process data in parallel
    pass
```

## Mejores Prácticas para RLPF

1. **Diseño del Modelo**:
   - Asegúrate de que tu modelo implemente todos los métodos requeridos de `BaseModel`
   - Utiliza el enmascaramiento adecuado para entradas de tamaño variable
   - Implementa la normalización y desnormalización adecuadas

2. **Función de Recompensa de Retroalimentación Física**:
   - Diseña recompensas que incorporen propiedades físicas significativas (p. ej., energía XTB, estabilidad)
   - Normaliza las recompensas para prevenir la inestabilidad del entrenamiento
   - Considera utilizar *reward shaping* para un mejor aprendizaje con restricciones físicas

3. **Proceso de Entrenamiento RLPF**:
   - Utiliza tamaños de `batch` adecuados para tu hardware
   - Monitorea las métricas de entrenamiento con `wandb`
   - Implementa un correcto guardado de checkpoints y del modelo
   - Equilibra la frecuencia de la retroalimentación física con el coste computacional

4. **Manejo de Datos**:
   - Usa `DataProto` para un intercambio de datos consistente
   - Implementa el `batching` y `chunking` adecuados
   - Maneja entradas de tamaño variable con el enmascaramiento adecuado

## Solución de Problemas

Problemas comunes y soluciones:

1. **Error de Memoria (Out of Memory)**:
   - Reduce el tamaño del `batch`
   - Utiliza acumulación de gradientes
   - Implementa una gestión de memoria adecuada

2. **Inestabilidad del Entrenamiento**:
   - Verifica la normalización de las recompensas
   - Ajusta la tasa de aprendizaje
   - Monitorea el recorte de ventaja (*advantage clipping*)

3. **Entrenamiento Lento**:
   - Utiliza tamaños de `batch` adecuados
   - Implementa procesamiento paralelo
   - Optimiza la carga de datos

## Contribuciones

¡Las contribuciones son bienvenidas! No dudes en enviar una Pull Request.

## Licencia

Este proyecto está licenciado bajo la Licencia Apache 2.0 - consulta el archivo LICENSE para más detalles.
