import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import time
from scipy.ndimage import gaussian_filter

class BradleyBinarization:
    def __init__(self, window_size=None, t=0.15):
        self.window_size = window_size
        self.t = t

    def binarize(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        height, width = image.shape

        if self.window_size is None:
            window_size = max(width // 8, height // 8)
        else:
            window_size = self.window_size

        integral = np.cumsum(np.cumsum(image.astype(np.float32), axis=0), axis=1)
        integral = np.pad(integral, ((1, 0), (1, 0)), mode='constant')

        binary = np.zeros_like(image)

        for y in range(height):
            for x in range(width):
                y1 = max(0, y - window_size // 2)
                x1 = max(0, x - window_size // 2)
                y2 = min(height - 1, y + window_size // 2)
                x2 = min(width - 1, x + window_size // 2)

                window_sum = (integral[y2 + 1, x2 + 1] - integral[y1, x2 + 1] -
                              integral[y2 + 1, x1] + integral[y1, x1])

                window_area = (y2 - y1 + 1) * (x2 - x1 + 1)
                mean = window_sum / window_area

                if image[y, x] < mean * (1 - self.t):
                    binary[y, x] = 0
                else:
                    binary[y, x] = 255

        return binary

class BradleyWithBackgroundCorrection:
    def __init__(self, window_size=None, t=0.15,
                 blur_kernel_size=31,
                 method='subtract',
                 correction_factor=128,
                 epsilon=1e-6):
        self.window_size = window_size
        self.t = t
        self.blur_kernel_size = blur_kernel_size
        self.method = method
        self.correction_factor = correction_factor
        self.epsilon = epsilon
        self.bradley = BradleyBinarization(window_size, t)

    def estimate_background(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sigma = self.blur_kernel_size / 6
        background = gaussian_filter(image.astype(np.float32), sigma=sigma)

        return background.astype(np.float32)

    def correct_illumination(self, image, background):
        if self.method == 'subtract':
            corrected = image.astype(np.float32) - background + self.correction_factor
            corrected = np.clip(corrected, 0, 255)

        elif self.method == 'divide':
            corrected = image.astype(np.float32) / (background + self.epsilon)
            corrected = np.clip(corrected * 255, 0, 255)

        return corrected.astype(np.uint8)

    def binarize(self, image):
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image.copy()

        background = self.estimate_background(image_gray)
        corrected_image = self.correct_illumination(image_gray, background)
        binary = self.bradley.binarize(corrected_image)

        return binary, corrected_image, background

class DatasetLoader:
    def __init__(self):
        self.processed_files = set()

    def load_dataset_from_folders(self, data_path):
        datasets = {
            'test': [],
            'train': [],
            'valid': []
        }

        subfolders = {
            'test': ['test', 'test_gt'],
            'train': ['train', 'train_gt'],
            'valid': ['valid', 'valid_gt']
        }

        for dataset_type, folders in subfolders.items():
            image_folder = folders[0]
            gt_folder = folders[1]

            image_path = os.path.join(data_path, image_folder)
            gt_path = os.path.join(data_path, gt_folder)

            if not os.path.exists(image_path):
                print(f"Ошибка: не найдена папка {image_path}")
                continue
            if not os.path.exists(gt_path):
                print(f"Ошибка: не найдена папка {gt_path}")
                continue

            print(f"Загрузка {dataset_type} данных...")

            image_files = []
            for root, dirs, files in os.walk(image_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                        full_path = os.path.join(root, file)
                        image_files.append(full_path)

            pairs = []
            for img_file in image_files:
                img_name = os.path.basename(img_file)
                gt_file = self._find_gt_file(img_name, gt_path)
                if gt_file:
                    pairs.append((img_file, gt_file))

            datasets[dataset_type] = pairs
            print(f"  Загружено пар: {len(pairs)}")

        return datasets

    def _find_gt_file(self, image_name, gt_folder):
        base_name = os.path.splitext(image_name)[0]

        patterns = [
            base_name,
            base_name + '_gt',
            base_name + '_GT',
            base_name + '_mask',
            base_name + '_Mask',
            base_name.replace('input', 'gt'),
            base_name.replace('Input', 'GT'),
            base_name.replace('image', 'mask'),
            base_name.replace('Image', 'Mask'),
            'gt_' + base_name,
            'GT_' + base_name,
            'mask_' + base_name,
            'Mask_' + base_name
        ]

        extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']

        for pattern in patterns:
            for ext in extensions:
                gt_path = os.path.join(gt_folder, pattern + ext)
                if os.path.exists(gt_path):
                    return gt_path

        for root, dirs, files in os.walk(gt_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                    gt_base = os.path.splitext(file)[0]
                    if (gt_base == base_name or
                            gt_base.startswith(base_name + '_') or
                            base_name.startswith(gt_base + '_')):
                        return os.path.join(root, file)

        return None

class MetricsCalculator:
    @staticmethod
    def calculate_confusion_matrix_elements(original, ground_truth):
        gt_binary = (ground_truth > 128).astype(np.uint8).flatten()
        result_binary = (original > 128).astype(np.uint8).flatten()

        cm = confusion_matrix(gt_binary, result_binary, labels=[0, 1])

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            if len(np.unique(gt_binary)) == 1 and len(np.unique(result_binary)) == 1:
                if gt_binary[0] == 0:
                    tn, fp, fn, tp = len(gt_binary), 0, 0, 0
                else:
                    tn, fp, fn, tp = 0, 0, 0, len(gt_binary)
            else:
                tn = np.sum((gt_binary == 0) & (result_binary == 0))
                fp = np.sum((gt_binary == 0) & (result_binary == 1))
                fn = np.sum((gt_binary == 1) & (result_binary == 0))
                tp = np.sum((gt_binary == 1) & (result_binary == 1))

        return tn, fp, fn, tp

    def evaluate_binarization(self, original, ground_truth):
        gt_binary = (ground_truth > 128).astype(np.uint8).flatten()
        result_binary = (original > 128).astype(np.uint8).flatten()

        precision = precision_score(gt_binary, result_binary, zero_division=0)
        recall = recall_score(gt_binary, result_binary, zero_division=0)
        f1 = f1_score(gt_binary, result_binary, zero_division=0)

        tn, fp, fn, tp = self.calculate_confusion_matrix_elements(original, ground_truth)

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'total_pixels': len(gt_binary)
        }

class ResultVisualizer:
    @staticmethod
    def visualize_correction_process_detailed(original, background, corrected,
                                              binary_classic, binary_corrected,
                                              ground_truth,
                                              classic_metrics=None,
                                              corrected_metrics=None):
        fig = plt.figure(figsize=(16, 9))

        gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.3, top=0.92, bottom=0.08)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original, cmap='gray')
        ax1.set_title('1. ИСХОДНОЕ ИЗОБРАЖЕНИЕ', fontsize=11, fontweight='bold', pad=10)
        ax1.text(0.5, -0.15, 'Входное полутоновое изображение',
                 transform=ax1.transAxes, ha='center', fontsize=8)
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(background, cmap='gray')
        ax2.set_title('2. ОЦЕНКА ФОНА', fontsize=11, fontweight='bold', pad=10)
        ax2.text(0.5, -0.15, 'Гауссово размытие (ядро=31)',
                 transform=ax2.transAxes, ha='center', fontsize=8)
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(corrected, cmap='gray')
        ax3.set_title('3. СКОРРЕКТИРОВАННОЕ', fontsize=11, fontweight='bold', pad=10)
        ax3.text(0.5, -0.15, 'I_corrected = I_original - I_background + C',
                 transform=ax3.transAxes, ha='center', fontsize=8)
        ax3.axis('off')

        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(ground_truth, cmap='gray')
        ax4.set_title('4. GROUND TRUTH', fontsize=11, fontweight='bold', pad=10)
        ax4.text(0.5, -0.15, 'Эталонная бинарная маска',
                 transform=ax4.transAxes, ha='center', fontsize=8)
        ax4.axis('off')

        ax5 = fig.add_subplot(gs[1, 0])
        ax5.imshow(binary_classic, cmap='gray')

        if classic_metrics:
            title_5 = f'5. КЛАССИЧЕСКИЙ БРЭДЛИ\nF1 = {classic_metrics["f1_score"]:.3f}'
        else:
            title_5 = '5. КЛАССИЧЕСКИЙ БРЭДЛИ'

        ax5.set_title(title_5, fontsize=11, fontweight='bold', pad=10)
        ax5.text(0.5, -0.15, 'Прямое применение алгоритма',
                 transform=ax5.transAxes, ha='center', fontsize=8)
        ax5.axis('off')

        ax6 = fig.add_subplot(gs[1, 1])
        ax6.imshow(binary_corrected, cmap='gray')

        if corrected_metrics:
            title_6 = f'6. БРЭДЛИ С КОРРЕКЦИЕЙ\nF1 = {corrected_metrics["f1_score"]:.3f}'
        else:
            title_6 = '6. БРЭДЛИ С КОРРЕКЦИЕЙ'

        ax6.set_title(title_6, fontsize=11, fontweight='bold', pad=10)
        ax6.text(0.5, -0.15, 'Алгоритм к скорректированному изображению',
                 transform=ax6.transAxes, ha='center', fontsize=8)
        ax6.axis('off')

        ax7 = fig.add_subplot(gs[1, 2])
        h, w = original.shape
        fragment_size = min(h, w) // 4
        start_h = (h - fragment_size) // 2
        end_h = start_h + fragment_size
        start_w = (w - fragment_size) // 2
        end_w = start_w + fragment_size

        ax7.imshow(original[start_h:end_h, start_w:end_w], cmap='gray')
        ax7.set_title('7. ФРАГМЕНТ: ИСХОДНОЕ', fontsize=10, fontweight='bold', pad=8)
        ax7.axis('off')

        ax8 = fig.add_subplot(gs[1, 3])
        comparison_fragment = np.zeros((fragment_size, fragment_size * 3, 3), dtype=np.uint8)

        classic_fragment = binary_classic[start_h:end_h, start_w:end_w]
        comparison_fragment[:, :fragment_size, 0] = classic_fragment
        comparison_fragment[:, :fragment_size, 1] = classic_fragment * 0.3
        comparison_fragment[:, :fragment_size, 2] = classic_fragment * 0.3

        corrected_fragment = binary_corrected[start_h:end_h, start_w:end_w]
        comparison_fragment[:, fragment_size:2 * fragment_size, 0] = corrected_fragment * 0.3
        comparison_fragment[:, fragment_size:2 * fragment_size, 1] = corrected_fragment
        comparison_fragment[:, fragment_size:2 * fragment_size, 2] = corrected_fragment * 0.3

        gt_fragment = ground_truth[start_h:end_h, start_w:end_w]
        comparison_fragment[:, 2 * fragment_size:3 * fragment_size, 0] = gt_fragment * 0.3
        comparison_fragment[:, 2 * fragment_size:3 * fragment_size, 1] = gt_fragment * 0.3
        comparison_fragment[:, 2 * fragment_size:3 * fragment_size, 2] = gt_fragment

        ax8.imshow(comparison_fragment)
        ax8.set_title('8. СРАВНЕНИЕ РЕЗУЛЬТАТОВ', fontsize=10, fontweight='bold', pad=8)
        ax8.axis('off')

        ax8.axvline(x=fragment_size - 0.5, color='white', linewidth=1, linestyle='--', alpha=0.7)
        ax8.axvline(x=2 * fragment_size - 0.5, color='white', linewidth=1, linestyle='--', alpha=0.7)

        if classic_metrics and corrected_metrics:
            improvement = corrected_metrics['f1_score'] - classic_metrics['f1_score']
            if improvement > 0:
                title_color = 'green'
                improvement_text = f'УЛУЧШЕНИЕ: ΔF1 = +{improvement:.3f}'
            else:
                title_color = 'red'
                improvement_text = f'УХУДШЕНИЕ: ΔF1 = {improvement:+.3f}'

            plt.suptitle(f'ПРОЦЕСС КОРРЕКЦИИ ОСВЕЩЕННОСТИ И БИНАРИЗАЦИИ\n{improvement_text}',
                         fontsize=7, fontweight='bold', y=0.98, color=title_color)
        else:
            plt.suptitle('ПРОЦЕСС КОРРЕКЦИИ ОСВЕЩЕННОСТИ И БИНАРИЗАЦИИ',
                         fontsize=7, fontweight='bold', y=0.98)

        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, hspace=0.3, wspace=0.3)
        plt.show()

        return

    @staticmethod
    def print_correction_analysis(original, background, corrected,
                                  classic_metrics, corrected_metrics):
        print()
        print("АНАЛИЗ ПРОЦЕССА КОРРЕКЦИИ ОСВЕЩЕННОСТИ")
        print()

        print("\n1. ЭТАПЫ ОБРАБОТКИ:")
        print("-" * 50)
        print("1. ИСХОДНОЕ ИЗОБРАЖЕНИЕ: Полутоновое изображение с возможными артефактами освещения")
        print("2. ОЦЕНКА ФОНА: Применение гауссова фильтра (σ≈5.17) для выделения низкочастотной составляющей")
        print("3. КОРРЕКЦИЯ: Вычитание оценки фона с добавлением константы C=128")
        print("4. БИНАРИЗАЦИЯ: Применение адаптивного алгоритма Брэдли к скорректированному изображению")

        print("\n2. КАЧЕСТВЕННЫЙ АНАЛИЗ:")
        print("-" * 50)

        if classic_metrics and corrected_metrics:
            print(f"Метрики качества для данного изображения:")
            print(f"  Классический Брэдли: F1 = {classic_metrics['f1_score']:.3f}, "
                  f"Precision = {classic_metrics['precision']:.3f}, "
                  f"Recall = {classic_metrics['recall']:.3f}")
            print(f"  Брэдли с коррекцией: F1 = {corrected_metrics['f1_score']:.3f}, "
                  f"Precision = {corrected_metrics['precision']:.3f}, "
                  f"Recall = {corrected_metrics['recall']:.3f}")

            improvement = corrected_metrics['f1_score'] - classic_metrics['f1_score']
            if improvement > 0:
                print(f"  УЛУЧШЕНИЕ: ΔF1 = {improvement:+.3f}")
            else:
                print(f"  УХУДШЕНИЕ: ΔF1 = {improvement:+.3f}")

        print("\n3. ВИЗУАЛЬНЫЕ НАБЛЮДЕНИИ:")
        print("-" * 50)

        original_mean = np.mean(original)
        corrected_mean = np.mean(corrected)

        print(f"Средняя интенсивность:")
        print(f"  Исходное изображение: {original_mean:.1f}")
        print(f"  Скорректированное:    {corrected_mean:.1f}")
        print(f"  Изменение:            {corrected_mean - original_mean:+.1f}")

        original_contrast = np.std(original)
        corrected_contrast = np.std(corrected)

        print(f"\nКонтраст (стандартное отклонение):")
        print(f"  Исходное изображение: {original_contrast:.1f}")
        print(f"  Скорректированное:    {corrected_contrast:.1f}")
        change_percent = ((corrected_contrast / original_contrast) - 1) * 100
        print(f"  Изменение:            {corrected_contrast - original_contrast:+.1f} ({change_percent:+.1f}%)")

        print("\n4. РЕКОМЕНДАЦИИ ДЛЯ ДАННОГО ИЗОБРАЖЕНИЯ:")
        print("-" * 50)

        if classic_metrics and corrected_metrics:
            if corrected_metrics['f1_score'] > classic_metrics['f1_score']:
                print("Коррекция освещенности улучшила качество бинаризации")
                print("Рекомендуется использовать модифицированный алгоритм")

                classic_error_rate = (classic_metrics['fp'] + classic_metrics['fn']) / classic_metrics['total_pixels']
                corrected_error_rate = (corrected_metrics['fp'] + corrected_metrics['fn']) / corrected_metrics[
                    'total_pixels']

                print(f"Общая ошибка снижена с {classic_error_rate:.2%} до {corrected_error_rate:.2%}")

            else:
                print("Коррекция не улучшила качество бинаризации")
                print("Для данного изображения достаточно классического алгоритма")
                print("Возможные причины: равномерное освещение или оптимальные исходные условия")

        print()

    @staticmethod
    def plot_confusion_matrices(classic_cm, corrected_cm, method_name="вычитания"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        sns.heatmap(classic_cm, annot=True, fmt=',d', cmap='Blues', ax=axes[0],
                    xticklabels=['Предсказан фон', 'Предсказан текст'],
                    yticklabels=['Реальный фон', 'Реальный текст'])

        axes[0].set_title('Классический метод Брэдли', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Предсказанный класс', fontsize=12)
        axes[0].set_ylabel('Реальный класс', fontsize=12)

        total_classic = classic_cm.sum()
        for i in range(2):
            for j in range(2):
                axes[0].text(j + 0.5, i + 0.3, f'{classic_cm[i, j] / total_classic:.1%}',
                             ha='center', va='center', fontsize=11, color='red')

        sns.heatmap(corrected_cm, annot=True, fmt=',d', cmap='Blues', ax=axes[1],
                    xticklabels=['Предсказан фон', 'Предсказан текст'],
                    yticklabels=['Реальный фон', 'Реальный текст'])

        axes[1].set_title(f'Брэдли с коррекцией {method_name}', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Предсказанный класс', fontsize=12)
        axes[1].set_ylabel('Реальный класс', fontsize=12)

        total_corrected = corrected_cm.sum()
        for i in range(2):
            for j in range(2):
                axes[1].text(j + 0.5, i + 0.3, f'{corrected_cm[i, j] / total_corrected:.1%}',
                             ha='center', va='center', fontsize=11, color='red')

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()

        diff_matrix = corrected_cm - classic_cm
        diff_percentage = (diff_matrix / total_classic) * 100

        print()
        print("АНАЛИЗ МАТРИЦ ОШИБОК")
        print()

        print("\nИзменения в матрице ошибок (Классический → Модифицированный):")
        print("-" * 60)
        print(f"True Negative (фон→фон):")
        print(f"  {classic_cm[0, 0]:,} → {corrected_cm[0, 0]:,} ({diff_percentage[0, 0]:+.1f}%)")

        print(f"\nFalse Positive (фон→текст):")
        print(f"  {classic_cm[0, 1]:,} → {corrected_cm[0, 1]:,} ({diff_percentage[0, 1]:+.1f}%)")

        print(f"\nFalse Negative (текст→фон):")
        print(f"  {classic_cm[1, 0]:,} → {corrected_cm[1, 0]:,} ({diff_percentage[1, 0]:+.1f}%)")

        print(f"\nTrue Positive (текст→текст):")
        print(f"  {classic_cm[1, 1]:,} → {corrected_cm[1, 1]:,} ({diff_percentage[1, 1]:+.1f}%)")

        print(f"\nАНАЛИЗ ИЗМЕНЕНИЙ:")
        print("-" * 60)

        if diff_percentage[0, 1] < 0:
            print(f"Уменьшение False Positive на {abs(diff_percentage[0, 1]):.1f}%")
            print("  Меньше фоновых пикселей ошибочно классифицировано как текст")
        else:
            print(f"Увеличение False Positive на {diff_percentage[0, 1]:.1f}%")
            print("  Больше фоновых пикселей ошибочно классифицировано как текст")

        if diff_percentage[1, 0] < 0:
            print(f"Уменьшение False Negative на {abs(diff_percentage[1, 0]):.1f}%")
            print("  Меньше текстовых пикселей ошибочно классифицировано как фон")
        else:
            print(f"Увеличение False Negative на {diff_percentage[1, 0]:.1f}%")
            print("  Больше текстовых пикселей ошибочно классифицировано как фон")

        print()

class BradleyComparisonTester:
    def __init__(self, datasets, num_samples=5, correction_method='subtract'):
        self.datasets = datasets
        self.num_samples = num_samples
        self.correction_method = correction_method

        self.metrics_calculator = MetricsCalculator()
        self.visualizer = ResultVisualizer()

        self.bradley_classic = BradleyBinarization(t=0.15)
        self.bradley_corrected = BradleyWithBackgroundCorrection(
            t=0.15,
            blur_kernel_size=31,
            method=correction_method,
            correction_factor=128,
            epsilon=1e-6
        )

        self.metrics_classic = {'test': [], 'train': [], 'valid': []}
        self.metrics_corrected = {'test': [], 'train': [], 'valid': []}
        self.processing_times_classic = {'test': [], 'train': [], 'valid': []}
        self.processing_times_corrected = {'test': [], 'train': [], 'valid': []}

        self.total_classic = {'test': {}, 'train': {}, 'valid': {}}
        self.total_corrected = {'test': {}, 'train': {}, 'valid': {}}

    def run_comparison(self, dataset_type='test'):
        print(f"\nЗапуск сравнения на {dataset_type} множестве...")
        print(f"Метод коррекции: {self.correction_method}")

        if not self.datasets or dataset_type not in self.datasets or not self.datasets[dataset_type]:
            print(f"Не найдено пар для обработки в множестве {dataset_type}")
            return None

        dataset = self.datasets[dataset_type]
        print(f"Найдено {len(dataset)} пар в множестве {dataset_type}")

        self._init_aggregated_metrics(dataset_type)

        processed_count = 0
        for i, (input_path, gt_path) in enumerate(dataset):

            result = self._process_single_comparison(processed_count, input_path, gt_path, dataset_type)
            if result:
                processed_count += 1

        return self._print_comparison_results(dataset_type)

    def _init_aggregated_metrics(self, dataset_type):
        for metric in ['tn', 'fp', 'fn', 'tp', 'total_pixels']:
            self.total_classic[dataset_type][metric] = 0
            self.total_corrected[dataset_type][metric] = 0

    def _process_single_comparison(self, index, input_path, gt_path, dataset_type):
        try:
            input_image = cv2.imread(input_path)
            ground_truth = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            if input_image is None or ground_truth is None:
                print(f"Ошибка загрузки: {os.path.basename(input_path)}")
                return False

            start_time = time.time()
            result_classic = self.bradley_classic.binarize(input_image)
            time_classic = time.time() - start_time
            self.processing_times_classic[dataset_type].append(time_classic)

            start_time = time.time()
            result_corrected, corrected_image, background = self.bradley_corrected.binarize(input_image)
            time_corrected = time.time() - start_time
            self.processing_times_corrected[dataset_type].append(time_corrected)

            score_classic = self.metrics_calculator.evaluate_binarization(result_classic, ground_truth)
            score_corrected = self.metrics_calculator.evaluate_binarization(result_corrected, ground_truth)

            self.metrics_classic[dataset_type].append(score_classic)
            self.metrics_corrected[dataset_type].append(score_corrected)

            for metric in ['tn', 'fp', 'fn', 'tp', 'total_pixels']:
                self.total_classic[dataset_type][metric] += score_classic[metric]
                self.total_corrected[dataset_type][metric] += score_corrected[metric]

            if (index + 1) % 10 == 0:
                print(f"{dataset_type.upper()} изображение {index + 1}:")
                print(f"  Классический: F1={score_classic['f1_score']:.3f}, Время: {time_classic:.3f} сек")
                print(f"  С коррекцией: F1={score_corrected['f1_score']:.3f}, Время: {time_corrected:.3f} сек")

            if dataset_type == 'test' and index < self.num_samples:
                self.visualizer.visualize_correction_process_detailed(
                    cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY),
                    background,
                    corrected_image,
                    result_classic,
                    result_corrected,
                    ground_truth,
                    classic_metrics=score_classic,
                    corrected_metrics=score_corrected
                )

                self.visualizer.print_correction_analysis(
                    cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY),
                    background,
                    corrected_image,
                    score_classic,
                    score_corrected
                )

            return True

        except Exception as e:
            print(f"Ошибка при обработке {input_path}: {e}")
            return False

    def _print_comparison_results(self, dataset_type):
        if not self.metrics_classic[dataset_type]:
            return None

        print()
        print(f"РЕЗУЛЬТАТЫ СРАВНЕНИЯ ({dataset_type.upper()} МНОЖЕСТВО)")
        print()
        print(f"Метод коррекции: {self.correction_method}")

        methods_data = [
            ("Классический Брэдли", self.metrics_classic[dataset_type],
             self.processing_times_classic[dataset_type], self.total_classic[dataset_type]),
            ("Брэдли с коррекцией", self.metrics_corrected[dataset_type],
             self.processing_times_corrected[dataset_type], self.total_corrected[dataset_type])
        ]

        results = {}

        for method_name, metrics_list, times_list, total_metrics in methods_data:
            avg_f1 = np.mean([m['f1_score'] for m in metrics_list])
            avg_precision = np.mean([m['precision'] for m in metrics_list])
            avg_recall = np.mean([m['recall'] for m in metrics_list])
            avg_time = np.mean(times_list)

            print(f"\n{method_name}:")
            print(f"  Обработано изображений: {len(metrics_list)}")
            print(f"  Средний F1-score:     {avg_f1:.4f}")
            print(f"  Средняя Precision:    {avg_precision:.4f}")
            print(f"  Средний Recall:       {avg_recall:.4f}")
            print(f"  Среднее время:        {avg_time:.3f} сек")
            print(f"  Общее время:          {sum(times_list):.2f} сек")

            accuracy = (total_metrics['tp'] + total_metrics['tn']) / total_metrics['total_pixels']
            print(f"  Общая Accuracy:       {accuracy:.4f}")

            print(f"\n  Матрица ошибок:")
            print(
                f"    True Negative:  {total_metrics['tn']:,} ({total_metrics['tn'] / total_metrics['total_pixels']:.2%})")
            print(
                f"    False Positive: {total_metrics['fp']:,} ({total_metrics['fp'] / total_metrics['total_pixels']:.2%})")
            print(
                f"    False Negative: {total_metrics['fn']:,} ({total_metrics['fn'] / total_metrics['total_pixels']:.2%})")
            print(
                f"    True Positive:  {total_metrics['tp']:,} ({total_metrics['tp'] / total_metrics['total_pixels']:.2%})")
            print(f"    Всего пикселей:  {total_metrics['total_pixels']:,}")

            results[method_name.lower().replace(' ', '_')] = {
                'avg_f1': avg_f1,
                'avg_precision': avg_precision,
                'avg_recall': avg_recall,
                'avg_time': avg_time,
                'accuracy': accuracy,
                'confusion_matrix': {
                    'tn': total_metrics['tn'],
                    'fp': total_metrics['fp'],
                    'fn': total_metrics['fn'],
                    'tp': total_metrics['tp']
                }
            }

        print()
        print(f"АНАЛИЗ РЕЗУЛЬТАТОВ СРАВНЕНИЯ")
        print()

        classic_f1 = results['классический_брэдли']['avg_f1']
        corrected_f1 = results['брэдли_с_коррекцией']['avg_f1']
        f1_improvement = corrected_f1 - classic_f1
        f1_improvement_percent = (f1_improvement / classic_f1 * 100) if classic_f1 > 0 else 0

        print(f"Изменение F1-score: {f1_improvement:+.4f} ({f1_improvement_percent:+.2f}%)")
        print(f"Классический метод: {classic_f1:.4f}")
        print(f"Модифицированный:   {corrected_f1:.4f}")

        classic_time = results['классический_брэдли']['avg_time']
        corrected_time = results['брэдли_с_коррекцией']['avg_time']
        time_increase = corrected_time - classic_time
        time_increase_percent = (time_increase / classic_time * 100) if classic_time > 0 else 0

        print(f"\nИзменение времени обработки:")
        print(f"Классический метод: {classic_time:.3f} сек")
        print(f"Модифицированный:   {corrected_time:.3f} сек")
        print(f"Изменение:          {time_increase:+.3f} сек ({time_increase_percent:+.1f}%)")

        if dataset_type == 'test':
            self._plot_confusion_matrices(results)

        if dataset_type == 'test':
            self._generate_correction_analysis_report(dataset_type)

        return results

    def _plot_confusion_matrices(self, results):
        classic_cm = np.array([
            [results['классический_брэдли']['confusion_matrix']['tn'],
             results['классический_брэдли']['confusion_matrix']['fp']],
            [results['классический_брэдли']['confusion_matrix']['fn'],
             results['классический_брэдли']['confusion_matrix']['tp']]
        ])

        corrected_cm = np.array([
            [results['брэдли_с_коррекцией']['confusion_matrix']['tn'],
             results['брэдли_с_коррекцией']['confusion_matrix']['fp']],
            [results['брэдли_с_коррекцией']['confusion_matrix']['fn'],
             results['брэдли_с_коррекцией']['confusion_matrix']['tp']]
        ])

        method_name = "вычитания" if self.correction_method == 'subtract' else "деления"

        print()
        print("ВИЗУАЛИЗАЦИЯ МАТРИЦ ОШИБОК")
        print()

        self.visualizer.plot_confusion_matrices(classic_cm, corrected_cm, method_name)

    def _generate_correction_analysis_report(self, dataset_type='test'):
        if not self.metrics_classic[dataset_type] or not self.metrics_corrected[dataset_type]:
            return

        print()
        print("ОТЧЕТ ПО АНАЛИЗУ ЭФФЕКТИВНОСТИ КОРРЕКЦИИ ОСВЕЩЕННОСТИ")
        print()

        total_images = len(self.metrics_classic[dataset_type])
        improved_count = 0
        worsened_count = 0
        unchanged_count = 0

        improvements = []

        for i in range(total_images):
            classic_f1 = self.metrics_classic[dataset_type][i]['f1_score']
            corrected_f1 = self.metrics_corrected[dataset_type][i]['f1_score']

            improvement = corrected_f1 - classic_f1
            improvements.append(improvement)

            if improvement > 0.01:
                improved_count += 1
            elif improvement < -0.01:
                worsened_count += 1
            else:
                unchanged_count += 1

        print(f"\nСТАТИСТИКА ПО {total_images} ИЗОБРАЖЕНИЯМ:")
        print("-" * 50)
        print(f"УЛУЧШЕНИЕ КАЧЕСТВА:    {improved_count} изображений ({improved_count / total_images * 100:.1f}%)")
        print(f"УХУДШЕНИЕ КАЧЕСТВА:    {worsened_count} изображений ({worsened_count / total_images * 100:.1f}%)")
        print(f"БЕЗ ЗНАЧИМЫХ ИЗМЕНЕНИЙ: {unchanged_count} изображений ({unchanged_count / total_images * 100:.1f}%)")

        avg_improvement = np.mean(improvements)
        median_improvement = np.median(improvements)
        std_improvement = np.std(improvements)

        print(f"\nРАСПРЕДЕЛЕНИЕ УЛУЧШЕНИЙ (ΔF1):")
        print("-" * 50)
        print(f"Среднее улучшение:    {avg_improvement:+.4f}")
        print(f"Медианное улучшение:  {median_improvement:+.4f}")
        print(f"Стандартное отклонение: {std_improvement:.4f}")
        print(f"Минимальное улучшение: {np.min(improvements):+.4f}")
        print(f"Максимальное улучшение: {np.max(improvements):+.4f}")

        fig_hist = plt.figure(figsize=(10, 6))
        ax_hist = fig_hist.add_subplot(111)

        n, bins, patches = ax_hist.hist(improvements, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax_hist.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Нет изменений')
        ax_hist.axvline(x=avg_improvement, color='green', linestyle='-', linewidth=2,
                        label=f'Среднее: {avg_improvement:+.3f}')

        ax_hist.set_xlabel('Улучшение F1-score (ΔF1)')
        ax_hist.set_ylabel('Количество изображений')
        ax_hist.set_title(f'Распределение улучшений качества на {dataset_type} множестве\n({total_images} изображений)')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)

        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
        plt.show()

        print(f"\nВЫВОДЫ И РЕКОМЕНДАЦИИ:")
        print()

        if avg_improvement > 0.01:
            print("Модифицированный алгоритм демонстрирует статистически значимое улучшение")
            print(f"Среднее улучшение F1-score: {avg_improvement:.3f}")
            print(f"Улучшение наблюдается на {improved_count / total_images * 100:.1f}% изображений")
            print("\nРекомендации:")
            print("1. Использовать модифицированный алгоритм для обработки документов")
            print("2. Особенно эффективен для изображений с неравномерным освещением")
            print("3. Может улучшить качество OCR за счет лучшей бинаризации")

        elif avg_improvement > 0:
            print("Модифицированный алгоритм показывает небольшое улучшение")
            print(f"Среднее улучшение F1-score: {avg_improvement:.3f}")
            print(f"Улучшение наблюдается на {improved_count / total_images * 100:.1f}% изображений")
            print("\nРекомендации:")
            print("1. Использовать модифицированный алгоритм в специфических случаях")
            print("2. Провести дополнительную настройку параметров коррекции")
            print("3. Рассмотреть другие методы предобработки изображений")

        else:
            print("Модифицированный алгоритм не улучшает качество бинаризации")
            print(f"Среднее изменение F1-score: {avg_improvement:.3f}")
            print(f"Ухудшение наблюдается на {worsened_count / total_images * 100:.1f}% изображений")
            print("\nРекомендации:")
            print("1. Использовать классический алгоритм Брэдли")
            print("2. Проанализировать причины неэффективности коррекции")
            print("3. Рассмотреть альтернативные подходы к компенсации освещения")

        print()

def main():
    data_path = r"C:\Users\User\.cache\kagglehub\datasets\lengocdatk16hcm\restomer-dibco-dataset\versions\1\content\Restomer_data"

    if not os.path.exists(data_path):
        print(f"Ошибка: путь {data_path} не существует!")
        return

    print()
    print("СИСТЕМА СРАВНЕНИЯ АЛГОРИТМОВ БИНАРИЗАЦИИ БРЭДЛИ")
    print()

    print("\nЗагрузка данных...")
    dataset_loader = DatasetLoader()
    datasets = dataset_loader.load_dataset_from_folders(data_path)

    if not datasets['test']:
        print("Ошибка: тестовое множество пусто!")
        return

    print(f"\nДанные успешно загружены:")
    print(f"  Train: {len(datasets['train'])} пар")
    print(f"  Valid: {len(datasets['valid'])} пар")
    print(f"  Test:  {len(datasets['test'])} пар")

    print()
    print("ТЕСТИРОВАНИЕ НА ТЕСТОВОМ МНОЖЕСТВЕ С МЕТОДОМ ВЫЧИТАНИЯ")
    print()
    tester_subtract = BradleyComparisonTester(datasets, num_samples=5, correction_method='subtract')
    results_subtract = tester_subtract.run_comparison('test')

    print()
    print("ТЕСТИРОВАНИЕ НА ТЕСТОВОМ МНОЖЕСТВЕ С МЕТОДОМ ДЕЛЕНИЯ")
    print()
    tester_divide = BradleyComparisonTester(datasets, num_samples=5, correction_method='divide')
    results_divide = tester_divide.run_comparison('test')

    if results_subtract and results_divide:
        print()
        print("ИТОГОВОЕ СРАВНЕНИЕ МЕТОДОВ КОРРЕКЦИИ")
        print()

        classic_f1 = results_subtract['классический_брэдли']['avg_f1']
        subtract_f1 = results_subtract['брэдли_с_коррекцией']['avg_f1']
        divide_f1 = results_divide['брэдли_с_коррекцией']['avg_f1']

        print(f"\nСредний F1-score на тестовом множестве:")
        print(f"  Классический Брэдли:          {classic_f1:.4f}")
        print(f"  С коррекцией (вычитание):     {subtract_f1:.4f} ({subtract_f1 - classic_f1:+.4f})")
        print(f"  С коррекцией (деление):       {divide_f1:.4f} ({divide_f1 - classic_f1:+.4f})")

        best_method = None
        best_improvement = 0

        if subtract_f1 > classic_f1:
            improvement = subtract_f1 - classic_f1
            if improvement > best_improvement:
                best_improvement = improvement
                best_method = "вычитание"

        if divide_f1 > classic_f1:
            improvement = divide_f1 - classic_f1
            if improvement > best_improvement:
                best_improvement = improvement
                best_method = "деление"

        if best_method:
            print(f"\nЛучший метод: {best_method} (улучшение на {best_improvement:.4f} F1-score)")
        else:
            print(f"\nЛучший метод: Классический Брэдли (без улучшений)")

        print(f"\nВременные характеристики:")
        print(f"  Классический:        {results_subtract['классический_брэдли']['avg_time']:.3f} сек")
        print(f"  Вычитание:           {results_subtract['брэдли_с_коррекцией']['avg_time']:.3f} сек")
        print(f"  Деление:             {results_divide['брэдли_с_коррекцией']['avg_time']:.3f} сек")

        print()
        print("ФИНАЛЬНЫЕ РЕКОМЕНДАЦИИ")
        print()

        if best_method == "вычитание":
            print("Рекомендуется использовать метод вычитания с параметрами:")
            print("  - Размер ядра Гауссова размытия: 31")
            print("  - Константа C: 128")
            print("  - Порог t: 0.15")
        elif best_method == "деление":
            print("Рекомендуется использовать метод деления с параметрами:")
            print("  - Размер ядра Гауссова размытия: 31")
            print("  - ε: 1e-6")
            print("  - Порог t: 0.15")
        else:
            print("Рекомендуется использовать классический метод Брэдли:")
            print("  - Порог t: 0.15")
            print("  - Размер окна: адаптивный (max(width, height) // 8)")

        print(f"\nИтоговое улучшение F1-score: {best_improvement:.4f}" if best_method else "\nУлучшений не обнаружено")

if __name__ == "__main__":
    main()