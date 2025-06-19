%% ================================================
% 0. Управление параметрами и автоматический выбор метода
% ================================================
% Здесь задаются все исходные параметры системы, а также
% – пороги и флаги, управляющие способом вычисления LCMV-весов.

% 0.1. Параметры «апертурного окна»
desired_sidelobe_atten_dB = 50;   % Желаемое подавление боковых лепестков (дБ)
                                    % (если <21, то beta=0, т.е. окно равно единице)

% 0.2. Параметры сигнала / времени
fc     = 25e9;                                   % Несущая частота (Гц)
lambda = physconst('LightSpeed') / fc;           % Длина волны (м)
fs     = 1e3;                                    % Частота дискретизации (Гц)
t      = (0 : 1/fs : 10)';                       % Временной вектор [0,10] сек

angle_az = 40;       % Целевая азимутальная координата (градусы)
angle_el = -30;      % Целевая координата угла места (градусы)

% 0.3. Параметры антенны URA (Uniform Rectangular Array)
M = 16;                 % Число элементов по вертикали (строк)
L = 16;                 % Число элементов по горизонтали (столбцов)
N = M * L;              % Общее число элементов решётки
d = lambda/2;           % Шаг между элементами (полуволны)

% 0.4. Пороги для автоматического включения методов:
opts.Woodbury_thresh = 10;     % Если K_total ≤ 10 → Woodbury
opts.Beamspace_frac  = 1/4;    % Если K_total ≥ N*(1/4) → Beamspace
opts.CG_maxIter      = 20;     % Число итераций для CG-решателя
opts.CG_tol          = 1e-6;   % Допустимый остаток для CG

% 0.5. Флаг загрузки (регуляризация ковариационной матрицы)
opts.adaptive_loading = true;   % true → epsDL = epsilon * trace(R)/N
opts.loading_epsilon  = 1e-3;   % Если adaptive_loading=false → epsDL = loading_epsilon

% 0.6. Флаг применения апертурного окна к uniform-весам
opts.WindowOnUniform = false;   % false → uniform-веса остаются равномерными 

% 0.7. Место под сигнал и шум
num_interferences       = 5;                % Число помех
interference_power_dB   = 15;               % Мощность каждой помехи (дБ)
noise_power            = 1;                % Мощность AWGN (единичная нормировка)
num_samples            = length(t);         % Число временных отсчётов

% 0.8. Флаги-результаты (будут переопределены далее)
opts.useWoodbury   = false;
opts.useBeamspace  = false;
opts.useCGsolver   = false;

%% ================================================
% 1. Создание антенной решётки URA (координаты элементов)
% ================================================
% Генерируем матрицу element_positions размером N×2:
% каждая строка — [xᵢ, yᵢ] координаты i-го элемента в плоскости XY.
element_positions = zeros(N, 2);
idx = 1;
for m_idx = 0:(M-1)
    for l_idx = 0:(L-1)
        x = l_idx * d;        % смещение по x 
        y = m_idx * d;        % смещение по y
        element_positions(idx, :) = [x, y];
        idx = idx + 1;
    end
end
% Теперь element_positions(i,:) — координаты i-го элемента URA.

%% ================================================
% 2. Генерация узкополосных помех и шума, оценка ковариации
% ================================================
% 2.1. Рандомизируем направления помех:
interference_angles_az = randperm(180, num_interferences) - 90;  % от −90° до +89°
interference_angles_el = randperm(180, num_interferences) - 90;  % от −90° до +89°

% 2.2. Предварительные steering-векторы помех
% Строим матрицу InterfVecs размера N×num_interferences
InterfVecs = zeros(N, num_interferences);
for i = 1:num_interferences
    az = interference_angles_az(i);
    el = interference_angles_el(i);
    InterfVecs(:, i) = compute_steering_vector(element_positions, el, az, lambda);
end

% 2.3. Генерация сигналов помех
% interference_signals — N×num_samples
% Каждая помеха узкополосная: sqrt(10^(interf_power/10))*exp(j*2π(f_c+1)*t)
interference_signals = zeros(N, num_samples);
for i = 1:num_interferences
    s_i = sqrt(10^(interference_power_dB/10)) * exp(1j*2*pi*(fc + 1) * t.');
    interference_signals = interference_signals + InterfVecs(:,i) * s_i;
end

% 2.4. Генерация белого гауссовского шума (AWGN)
noise = sqrt(noise_power/2) * (randn(N, num_samples) + 1j*randn(N, num_samples));

% 2.5. Суммируем помехи + шум → получаем принятый N×num_samples
received_signal = interference_signals + noise;

% 2.6. Строим выборочную ковариационную матрицу R = (X * X^H) / num_samples
R = (received_signal * received_signal') / num_samples;
% Эта R содержит информацию о шуме + помехах на каждом элементе URA.

%% ================================================
% 3. Адаптивная диагональная загрузка (Regularization)
% ================================================
[U_R, S_R, V_R] = svd(R);                      % SVD-разложение R = U_R * S_R * V_R'
if opts.adaptive_loading
    epsDL = opts.loading_epsilon * trace(R) / N;   % адаптивно относительно trace(R)
else
    epsDL = opts.loading_epsilon;               % фиксированная диагональная загрузка
end
R_reg = R + epsDL * eye(N);                    % Регуляризованная ковариация

% Показываем, как поменялось число обусловленности:
cond_R_before = cond(S_R);
[~, S_R2, ~] = svd(R_reg);
cond_R_after  = cond(S_R2);
disp(['cond(R) до = ',  num2str(cond_R_before,'%.2e'), ...
      ', cond(R\_reg) после = ', num2str(cond_R_after,'%.2e')]);

%% ================================================
% 4. Апертура́нное окно Кайзера (2D)
% ================================================
% 4.1. Расчёт параметра β в зависимости от desired_sidelobe_atten_dB
if desired_sidelobe_atten_dB > 50
    beta_kaiser = 0.1102 * (desired_sidelobe_atten_dB - 8.7);
elseif desired_sidelobe_atten_dB >= 21
    beta_kaiser = 0.5842 * (desired_sidelobe_atten_dB - 21)^0.4 ...
                  + 0.07886 * (desired_sidelobe_atten_dB - 21);
else
    beta_kaiser = 0;  % это приводит к единичному окну
end

% 4.2. Генерация одногоерного окна по вертикали (M×1) и горизонтали (L×1)
w_vert  = my_kaiser(M, beta_kaiser);   % M×1
w_horiz = my_kaiser(L, beta_kaiser);   % L×1

% 4.3. Формируем двумерное окно W2D = w_vert * w_horiz'
W2D = w_vert * w_horiz.';  % размер M×L

% 4.4. Разворачиваем вектором: w_2d размер N×1, где N = M*L
w_2d = W2D(:);            
w_2d = w_2d / norm(w_2d);  % нормируем, чтобы ||w_2d|| = 1

%% ================================================
% 5. Построение матрицы ограничений A и вектора g
% ================================================
% 5.1. Целевая steering-вектор для (angle_el, angle_az)
Steer_target = compute_steering_vector(element_positions, angle_el, angle_az, lambda);

% 5.2. Matрица A размером N×(1+num_interferences):
% Первый столбец — Steer_target, далее — InterfVecs
A = [Steer_target, InterfVecs]; 
% Общее число столбцов = K_total = 1 + num_interferences

% 5.3. Вектор g размером (1+num_interferences)×1:
% Желательный отклик: в направлении цели — 1, в направлениях помех — 0
g = [1; zeros(num_interferences,1)];

% Запомним K_total
K_total = size(A, 2);

%% ================================================
% 6. Автоматический выбор метода расчёта LCMV-весов
% ================================================
% Логика:
%  - если K_total ≤ Woodbury_thresh → включаем Woodbury
%  - иначе, если K_total ≥ N*(Beamspace_frac) → Beamspace
%  - иначе → CG-решатель
if K_total <= opts.Woodbury_thresh
    opts.useWoodbury   = true;
    opts.useBeamspace  = false;
    opts.useCGsolver   = false;
    disp('→ Выбрано: Woodbury (K_total ≤ 10)');
    
elseif K_total >= ceil(N * opts.Beamspace_frac)
    opts.useWoodbury   = false;
    opts.useBeamspace  = true;
    opts.useCGsolver   = false;
    disp('→ Выбрано: Beamspace (K_total ≥ N/4)');
    
else
    opts.useWoodbury   = false;
    opts.useBeamspace  = false;
    opts.useCGsolver   = true;
    disp('→ Выбрано: CG-решатель (10 < K_total < N/4)');
end

%% ================================================
% 7. Вычисление LCMV-весов w_lcmv с учётом выбранного метода
% ================================================
if opts.useWoodbury
    %––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    % 7.1. Woodbury (“матрица ограничений небольшая” K_total ≤ 10)
    % Если R_reg = U_R * (S_R + epsDL*I) * V_R', то:
    % R_reg^{-1} = U_R * diag(1./(diag(S_R)+epsDL)) * U_R'
    Sigma_reg = diag(1 ./ (diag(S_R) + epsDL));  
    Rinv = U_R * Sigma_reg * U_R';     % N×N matrix-inverse

    RA = Rinv * A;                     % N×K_total
    G  = A' * RA;                      % K_total×K_total (Gram-матрица ограничений)
    % w_lcmv = RA * (G \ g)            % аналогично (Rinv*A)*( (A'*Rinv*A)^{-1} * g )
    w_lcmv = RA * (G \ g);             % N×1
    disp('— LCMV через Woodbury');

elseif opts.useBeamspace
    %––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    % 7.2. Beamspace (K_total очень велико, ≥ N/4)
    %  Проецируем R_reg и A в подпространство размерности B = round(N/4)
    B = ceil(N * opts.Beamspace_frac);
    idx_n = (0:(N-1))';                % N×1
    idx_b = 0:(B-1);                   % 1×B
    % Собираем матрицу D размером N×B — первые B столбцов DFT (нормализация 1/√N)
    D = exp(-1j*(2*pi/N)*(idx_n * idx_b)) / sqrt(N);
    % Производим фазово-пространственную проекцию:
    Rb  = D' * R_reg * D;              % B×B
    A_b = D' * A;                      % B×K_total

    % Теперь решаем «маленький» LCMV в пространстве B:
    w_b = (Rb \ A_b) / (A_b' * (Rb \ A_b)) * g;  % B×1
    % Проецируем обратно в N-мерное пространство:
    w_lcmv = D * w_b;                  % N×1
    disp('— LCMV через Beamspace');

elseif opts.useCGsolver
    %––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    % 7.3. CG-решатель (промежуточное K_total)
    % Решаем R_reg * X = A итеративно по столбцам
    K_tot = size(A,2);
    X     = zeros(N, K_tot);
    tol   = opts.CG_tol;
    maxIt = opts.CG_maxIter;
    for col = 1:K_tot
        b = A(:,col);
        x0 = zeros(N,1);
        [x_cg, ~] = my_CG_solver(R_reg, b, x0, maxIt, tol);
        X(:,col) = x_cg;
    end
    G       = A' * X;                  % K_total × K_total
    w_lcmv  = X * (G \ g);             % N×1
    disp('— LCMV через CG-решатель');

else
    %––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    % 7.4. Классический прямой метод (по умолчанию, если K очень мал или не попало ни под Woodbury, ни под Beamspace, ни под CG)
    w_lcmv = (R_reg \ A) / (A' * (R_reg \ A)) * g;  % N×1
    disp('— LCMV через прямой метод');
end

%% ================================================
% 8. Предвычисление steering-векторов по всему угловому гриду
% ================================================
% Диаграммы направленности будут строиться на гриде:
azimuths = -90:1:90;                     % –90° … +90° с шагом 1°
elevations= -90:1:90;                    % –90° … +90° с шагом 1°
[AZ, EL] = meshgrid(azimuths, elevations);  % AZ, EL — matrices размером 181×181
P = numel(AZ);                             % Всего точек = 181×181

% Предвычисляем все steering-векторы размером N×P:
SteerAll = zeros(N, P);
% Цикл по всем азимутам/углам места
for i = 1:P
    SteerAll(:,i) = compute_steering_vector(element_positions, EL(i), AZ(i), lambda);
end

%% ================================================
% 9. Расчёт диаграмм направленности
% ================================================
% 9.1. LCMV-диаграмма (c апертурным окном w_2d)
ApodSteer_LCMV = SteerAll .* (w_2d * ones(1,P));  % N×P
resp_lcmv = w_lcmv' * ApodSteer_LCMV;             % 1×P комплексный вектор откликов
% Dвн: 10·log10(|resp|^2 / (w^H * w))
pat_lcmv = 10*log10(abs(resp_lcmv).^2 / (w_lcmv' * w_lcmv));
% Нормировка по максимуму (максимум = 0 дБ)
pat_lcmv = pat_lcmv - max(pat_lcmv);
% «Обрезаем» уровни ниже −60 дБ
pat_lcmv(pat_lcmv < -60) = -60;
pattern_lcmv = reshape(pat_lcmv, size(AZ));       % 181×181

% 9.2. Uniform-диаграмма (равномерные веса), окно не накладываем, если opts.WindowOnUniform=false
W_uniform = ones(N,1) / sqrt(N);  % равномерные веса (N×1)
if opts.WindowOnUniform
    %  Если нужно применить апертурное окно к uniform
    W_uniform = (w_2d .* W_uniform);
    W_uniform = W_uniform / norm(W_uniform);
end
ApodSteer_UNF = SteerAll .* (ones(N,1) * ones(1,P));  
%  Если opts.WindowOnUniform=false, то фактически ApodSteer_UNF = SteerAll
resp_unif = W_uniform' * ApodSteer_UNF;  % 1×P
pat_unif = 10*log10(abs(resp_unif).^2 / (W_uniform' * W_uniform));
pat_unif = pat_unif - max(pat_unif);
pat_unif(pat_unif < -60) = -60;
pattern_uniform = reshape(pat_unif, size(AZ));    % 181×181

%% ================================================
% 10. Визуализация результатов (2D и 3D)
% ================================================
figure('Name', 'Диограммы направленности с авт. выбором метода', 'Position', [100,100,1400,900]);

% 10.1. LCMV (2D)
subplot(2,2,1);
imagesc(azimuths, elevations, pattern_lcmv);
axis xy;
colorbar;
colormap(turbo);
title('LCMV (2D)');
xlabel('Азимут (°)');
ylabel('Угол места (°)');
caxis([-60 0]);
grid on; hold on;
% Контуры каждые 10 дБ от −60 до 0
contour(AZ, EL, pattern_lcmv, -60:10:0, 'LineColor', 'k');
% Маркеры направлений помех (белые крестики)
plot(interference_angles_az, interference_angles_el, 'wx', 'LineWidth', 3, 'MarkerSize', 10);

% 10.2. Uniform (2D)
subplot(2,2,2);
imagesc(azimuths, elevations, pattern_uniform);
axis xy;
colorbar;
colormap(turbo);
title('Без beamforming (2D)');
xlabel('Азимут (°)');
ylabel('Угол места (°)');
caxis([-60 0]);
grid on; hold on;
contour(AZ, EL, pattern_uniform, -60:10:0, 'LineColor', 'k');
plot(interference_angles_az, interference_angles_el, 'wx', 'LineWidth', 3, 'MarkerSize', 10);

% 10.3. LCMV (3D)
subplot(2,2,3);
surf(AZ, EL, pattern_lcmv, 'EdgeColor', 'none');
shading interp;
view(45, 30);
title('LCMV (3D)');
xlabel('Азимут (°)');
ylabel('Угол места (°)');
zlabel('Усиление (дБ)');
colorbar;
colormap(turbo);
caxis([-60 0]);
hold on;
% Помехи на уровне −60 дБ (белые кресты в трёхмерных координатах)
plot3(interference_angles_az, interference_angles_el, -60*ones(size(interference_angles_az)), 'wx', 'MarkerSize', 8, 'LineWidth', 2);

% 10.4. Uniform (3D)
subplot(2,2,4);
surf(AZ, EL, pattern_uniform, 'EdgeColor', 'none');
shading interp;
view(45, 30);
title('Без beamforming (3D)');
xlabel('Азимут (°)');
ylabel('Угол места (°)');
zlabel('Усиление (дБ)');
colorbar;
colormap(turbo);
caxis([-60 0]);
hold on;
plot3(interference_angles_az, interference_angles_el, -60*ones(size(interference_angles_az)), 'wx', 'MarkerSize', 8, 'LineWidth', 2);

%% ================================================
% 11. Вспомогательные функции
% ================================================
function a = compute_steering_vector(positions, theta_deg, phi_deg, lambda)
    % compute_steering_vector: 
    %   Формирует steering-вектор для URA (N×1). 
    % Вход:
    %   positions — N×2 массив: [xᵢ, yᵢ] координаты каждого элемента
    %   theta_deg   — угол места (в градусах)
    %   phi_deg     — азимут (в градусах)
    %   lambda      — длина волны
    % Выход:
    %   a — N×1 комплексный steering-вектор
    theta = deg2rad(theta_deg);
    phi   = deg2rad(phi_deg);
    k     = 2 * pi / lambda;                                 % волновое число
    dir   = [sin(theta)*cos(phi), sin(theta)*sin(phi)];      % проекции вектора направления
    % a(i) = exp(j * k * (xᵢ*dir(1) + yᵢ*dir(2)))
    a = exp(1j * k * (positions * dir.'));  % (N×2)*(2×1) → N×1
end

function w = my_kaiser(N, beta)
    % my_kaiser:
    %   Приближённая реализация Kaiser-окна длины N с параметром beta
    %   Аппроксимация I0(x) делается суммой ряда Маклорена до K=20.
    %
    % Вход:
    %   N    — длина окна
    %   beta — параметр (чем больше beta, тем сильнее подавляются боковые лепестки)
    % Выход:
    %   w — N×1 вектор Kaiser-окна, нормированного так, что w(alpha)=1
    %
    K = 20;                     % число членов ряда Маклорена
    % 1) Считаем нормирующий множитель I0(beta):
    I0_beta = 0;
    for k_i = 0:K
        I0_beta = I0_beta + ((0.5 * beta)^(2 * k_i)) / (factorial(k_i)^2);
    end
    alpha = (N - 1) / 2;
    w     = zeros(N, 1);
    for n = 0:(N - 1)
        ratio = (n - alpha) / alpha;
        arg   = beta * sqrt(1 - ratio^2);
        % 2) Считаем I0(arg):
        I0_arg = 0;
        for k_i = 0:K
            I0_arg = I0_arg + ((0.5 * arg)^(2 * k_i)) / (factorial(k_i)^2);
        end
        w(n + 1) = I0_arg / I0_beta;   % нормируем, чтобы w(alpha)=1
    end
end

function [x, iter] = my_CG_solver(A, b, x0, maxIter, tol)
    % my_CG_solver:
    %   Решает систему A*x = b (A Hermitian, Positive-Definite) методом сопряжённых градиентов.
    % Вход:
    %   A       — матрица N×N (неявно Hermitian PD, мы используем R_reg)
    %   b       — вектор N×1
    %   x0      — начальное приближение N×1 (обычно ноль)
    %   maxIter — максимальное число итераций
    %   tol     — допускаемый остаток (критерий останова)
    % Выход:
    %   x    — решение N×1
    %   iter — число выполненных итераций
    x    = x0;
    r    = b - A*x;     % начальный остаток
    p    = r;
    rsold = r' * r;
    for iter = 1:maxIter
        Ap = A * p;
        alpha = rsold / (p' * Ap);
        x = x + alpha * p;
        r = r - alpha * Ap;
        rsnew = r' * r;
        if sqrt(rsnew) < tol
            break;
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    end
end
