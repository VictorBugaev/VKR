%% ================================ % 0. Управление параметрами % ================================
desired_sidelobe_atten_dB = 30;  % Желаемое подавление боковых лепестков (дБ)

%% ================================ % 1. Параметры системы и создание антенной решетки % ================================
fc = 25e9;  % Частота (Гц)
lambda = physconst('LightSpeed') / fc;  % Длина волны
fs = 1e3;  % Частота дискретизации (Гц)
t = (0:1/fs:10)';  % Вектор времени от 0 до 10 с шагом 1/fs
angle_az = 40;  % Азимут (°)
angle_el = -30;  % Угол места (°)

M = 16;  % Количество элементов по вертикали
L = 16;  % Количество элементов по горизонтали
N = M * L;  % Общее количество элементов
d = lambda / 2;  % Расстояние между элементами (м)

% Координаты элементов URA (фар)
element_positions = zeros(N, 2);
index = 1;
for m = 0:M-1
    for l = 0:L-1
        x = l * d;
        y = m * d;
        element_positions(index, :) = [x, y];
        index = index + 1;
    end
end

%% ================================ % 2. Генерация сигналов помех и шума % ================================
num_interferences = 5; % Количество помех
interference_angles_az = randperm(180, num_interferences) - 90;
interference_angles_el = randperm(180, num_interferences) - 90;
interference_power = 15;  % дБ
noise_power = 1;  % Нормированная мощность шума
num_samples = length(t);

% Генерация узкополосных помех
interference_signals = zeros(N, num_samples);
for i = 1:num_interferences
    az = interference_angles_az(i);
    el = interference_angles_el(i);
    steering_vec = compute_steering_vector(element_positions, el, az, lambda);
    freq_offset = 1; % Смещение частоты помехи (Гц)
    s_i = sqrt(10^(interference_power/10)) * exp(1j*2*pi*(fc + freq_offset)*t.');
    interference_signals = interference_signals + steering_vec * s_i;
end

% Добавление шума
noise = sqrt(noise_power/2)*(randn(N, num_samples) + 1j*randn(N, num_samples));
received_signal = interference_signals + noise;

R = (received_signal * received_signal') / num_samples;

%% ================================ % 3. Адаптивный выбор окна Кайзера % ================================
if desired_sidelobe_atten_dB > 50
    beta_kaiser = 0.1102 * (desired_sidelobe_atten_dB - 8.7);
elseif desired_sidelobe_atten_dB >= 21
    beta_kaiser = 0.5842 * (desired_sidelobe_atten_dB - 21)^0.4 + 0.07886 * (desired_sidelobe_atten_dB - 21);
else
    beta_kaiser = 0;
end
disp(['Используем β окна Кайзера: ', num2str(beta_kaiser, '%.2f')]);

w_vert = my_kaiser(M, beta_kaiser);
w_horiz = my_kaiser(L, beta_kaiser);
w_2d = w_vert * w_horiz.';  % Двумерное окно
w_2d = w_2d(:);
w_2d = w_2d / norm(w_2d);  % Нормализация

%% ================================ % 4. Построение ограничений для LCMV % ================================
target_az = angle_az;
target_el = angle_el;
interference_az = interference_angles_az;
interference_el = interference_angles_el;

directions_deg = [ [target_el, target_az]; [interference_el(:), interference_az(:)] ];
A = zeros(N, size(directions_deg,1));
for i = 1:size(directions_deg,1)
    A(:,i) = compute_steering_vector(element_positions, directions_deg(i,1), directions_deg(i,2), lambda);
end
g = [1; zeros(size(directions_deg,1)-1,1)];

%% ================================ % 5. Расчет весов LCMV с уменьшением регуляризации % ================================
[U, S, V] = svd(R);
cond_R = cond(S);
delta = max(diag(S));
disp(['Условное число R: ', num2str(cond_R, '%.2e'), ', выбран delta: ', num2str(delta)]);

epsilon = 1e-3;
R_reg = R + epsilon * trace(R)/N * eye(N);
weights_lcmv = (R_reg \ A) / (A' * (R_reg \ A)) * g;

%% ================================ % 6. Расчет диаграмм направленности % ================================
azimuths = -90:1:90;
elevations = -90:1:90;
[AZ, EL] = meshgrid(azimuths, elevations);

pattern_lcmv = zeros(size(AZ));
pattern_uniform = zeros(size(AZ));

w_uniform = ones(N, 1) / sqrt(N);

for i = 1:numel(AZ)
    az = AZ(i);
    el = EL(i);
    a_theta = compute_steering_vector(element_positions, el, az, lambda);

    a_theta_lcmv = a_theta .* w_2d;
    response_lcmv = weights_lcmv' * a_theta_lcmv;
    directivity_lcmv = (abs(response_lcmv)^2) / (weights_lcmv' * weights_lcmv);
    pattern_lcmv(i) = 10 * log10(directivity_lcmv);

    a_theta_uniform = a_theta .* w_2d;
    response_uniform = w_uniform' * a_theta_uniform;
    directivity_uniform = (abs(response_uniform)^2) / (w_uniform' * w_uniform);
    pattern_uniform(i) = 10 * log10(directivity_uniform);
end

pattern_lcmv = pattern_lcmv - max(pattern_lcmv(:));
pattern_uniform = pattern_uniform - max(pattern_uniform(:));

pattern_lcmv(pattern_lcmv < -60) = -60;
pattern_uniform(pattern_uniform < -60) = -60;

%% ================================ % 7. Построение 2D и 3D карт % ================================
figure('Name', 'Диограммы направленности', 'Position', [100, 100, 1400, 900]);

subplot(2, 2, 1);
imagesc(azimuths, elevations, pattern_lcmv);
axis xy;
colorbar;
colormap(turbo);
title('LCMV (2D)');
xlabel('Азимут (°)');
ylabel('Угол места (°)');
caxis([-60 0]);
grid on;
hold on;
contour(AZ, EL, pattern_lcmv, -60:10:0, 'LineColor', 'k');
plot(interference_az, interference_el, 'wx', 'LineWidth', 3, 'MarkerSize', 10);

subplot(2, 2, 2);
imagesc(azimuths, elevations, pattern_uniform);
axis xy;
colorbar;
colormap(turbo);
title('Без beamforming (2D)');
xlabel('Азимут (°)');
ylabel('Угол места (°)');
caxis([-60 0]);
grid on;
hold on;
contour(AZ, EL, pattern_uniform, -60:10:0, 'LineColor', 'k');
% Добавим маркеры направлений помех
plot(interference_az, interference_el, 'wx', 'LineWidth', 3, 'MarkerSize', 10);

subplot(2, 2, 3);
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

subplot(2, 2, 4);
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

%% ================================ % 8. Реалтайм-анимация перемещения цели ================================
% num_steps = 20;
% target_az_traj = 45 + 15 * sin(2 * pi * (1:num_steps) / num_steps);
% target_el_traj = 45 + 10 * cos(2 * pi * (1:num_steps) / num_steps);
% 
% figure('Name', 'Адаптация диаграммы в реальном времени', 'Position', [100, 100, 1400, 900]);
% for k = 1:num_steps
%     target_az = target_az_traj(k);
%     target_el = target_el_traj(k);
% 
%     steering_target = compute_steering_vector(element_positions, target_el, target_az, lambda);
%     A_dyn = zeros(N, 1 + num_interferences);
%     g_dyn = zeros(1 + num_interferences, 1);
%     A_dyn(:,1) = steering_target;
%     g_dyn(1) = 1;
%     for i = 1:num_interferences
%         A_dyn(:,i+1) = compute_steering_vector(element_positions, interference_el(i), interference_az(i), lambda);
%     end
% 
%     weights_dyn = (R_reg \ A_dyn) / (A_dyn' * (R_reg \ A_dyn)) * g_dyn;
% 
%     pattern_rt = zeros(size(AZ));
%     for i = 1:numel(AZ)
%         az = AZ(i);
%         el = EL(i);
%         a_theta = compute_steering_vector(element_positions, el, az, lambda);
%         response = weights_dyn' * (a_theta .* w_2d);
%         pattern_rt(i) = 10 * log10(abs(response)^2);
%     end
% 
%     pattern_rt = pattern_rt - max(pattern_rt(:));
%     pattern_rt(pattern_rt < -60) = -60;
% 
%     subplot(1,2,1);
%     imagesc(azimuths, elevations, pattern_rt);
%     axis xy; colormap(turbo); caxis([-60 0]); colorbar;
%     title(sprintf('2D: шаг %d, азимут %.1f°, угол %.1f°', k, target_az, target_el));
%     xlabel('Азимут (°)'); ylabel('Угол места (°)');
% 
%     subplot(1,2,2);
%     surf(AZ, EL, pattern_rt, 'EdgeColor', 'none');
%     shading interp; view(45, 30);
%     title('3D'); xlabel('Азимут'); ylabel('Угол места'); zlabel('Усиление (дБ)');
%     colorbar; colormap(turbo); caxis([-60 0]);
% 
%     drawnow;
%     pause(0.5);
% end

%% ================================ % 9. Вычисление steering-вектора % ================================
function a = compute_steering_vector(positions, theta_deg, phi_deg, lambda)
    theta = deg2rad(theta_deg);
    phi = deg2rad(phi_deg);
    k = 2 * pi / lambda;
    dir = [sin(theta) * cos(phi), sin(theta) * sin(phi)];
    a = exp(1j * k * (positions * dir.'));  % векторизованный скалярный продукт
end

%% ================================ % 10. Функция окна Кайзера % ================================
function w = my_kaiser(N, beta)
    % Аппроксимация I0(x) через ряд Маклорена с использованием factorial
    K = 20;               % число членов ряда
    w = zeros(N, 1);

    % Предвычисляем нормирующий множитель I0(beta)
    I0_beta = 0;
    for k = 0:K
        term = (0.5 * beta)^(2*k) / (factorial(k)^2);
        I0_beta = I0_beta + term;
    end

    alpha = (N - 1) / 2;
    for n = 0:N-1
        ratio = (n - alpha) / alpha;
        arg = beta * sqrt(1 - ratio^2);

        % Аппроксимация I0(arg)
        I0_arg = 0;
        for k = 0:K
            term = (0.5 * arg)^(2*k) / (factorial(k)^2);
            I0_arg = I0_arg + term;
        end

        w(n+1) = I0_arg / I0_beta;  % нормируем
    end
end