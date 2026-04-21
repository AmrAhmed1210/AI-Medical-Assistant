using AutoMapper;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.UserModule;
using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Domain.Entities.AppointmentsModule;
using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Domain.Entities.SessionsModule;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.Admin;
using MedicalAssistant.Shared.DTOs.Common;

namespace MedicalAssistant.Services.Services
{
    public class AdminService : IAdminService
    {
        private readonly IUnitOfWork _unitOfWork;
        private readonly IMapper _mapper;
        private readonly INotificationService _notificationService;

        public AdminService(IUnitOfWork unitOfWork, IMapper mapper, INotificationService notificationService)
        {
            _unitOfWork = unitOfWork;
            _mapper = mapper;
            _notificationService = notificationService;
        }

        public async Task<SystemStatsDto> GetSystemStatsAsync()
        {
            try
            {
                var users = await _unitOfWork.Repository<User>().GetAllAsync();
                var patients = await _unitOfWork.Patients.GetAllAsync();
                var appointments = await _unitOfWork.Repository<Appointment>().GetAllAsync();
                
                List<Session> sessions = new();
                try 
                {
                    sessions = (await _unitOfWork.Repository<Session>().GetAllAsync()).ToList();
                }
                catch 
                {
                    // Sessions table might be empty or not exist yet
                    sessions = new List<Session>();
                }
                
                var today = DateTime.UtcNow.Date;
                var weekAgo = today.AddDays(-7);
                var sixMonthsAgo = today.AddMonths(-6);

                var activeUsers = users.Where(u => !u.IsDeleted).ToList();
                var activePatients = patients.Where(p => p.IsActive).ToList();

                // Urgency distribution from all sessions
                var urgencyGroups = sessions
                    .Where(s => !string.IsNullOrEmpty(s.UrgencyLevel))
                    .GroupBy(s => s.UrgencyLevel!)
                    .ToDictionary(g => g.Key, g => g.Count());

                var urgencyDistribution = new Dictionary<string, int>
                {
                    ["LOW"] = urgencyGroups.GetValueOrDefault("LOW", 0),
                    ["MEDIUM"] = urgencyGroups.GetValueOrDefault("MEDIUM", 0),
                    ["HIGH"] = urgencyGroups.GetValueOrDefault("HIGH", 0),
                    ["EMERGENCY"] = urgencyGroups.GetValueOrDefault("EMERGENCY", 0)
                };

                // Sessions per day (last 7 days)
                var sessionsPerDay = sessions
                    .Where(s => s.CreatedAt.Date >= weekAgo)
                    .GroupBy(s => s.CreatedAt.Date)
                    .Select(g => new DateCountDto 
                    { 
                        Date = g.Key.ToString("yyyy-MM-dd"), 
                        Count = g.Count() 
                    })
                    .OrderBy(x => x.Date)
                    .ToList();

                // User growth (last 6 months) - cumulative count by month
                var userGrowth = activeUsers
                    .Where(u => u.CreatedAt >= sixMonthsAgo)
                    .GroupBy(u => new { u.CreatedAt.Year, u.CreatedAt.Month })
                    .Select(g => new DateCountDto
                    {
                        Date = new DateTime(g.Key.Year, g.Key.Month, 1).ToString("MMM"),
                        Count = g.Count()
                    })
                    .OrderBy(x => x.Date)
                    .ToList();

                return new SystemStatsDto
                {
                    TotalUsers = activeUsers.Count() + activePatients.Count(),
                    TotalDoctors = activeUsers.Count(u => u.Role == "Doctor"),
                    TotalPatients = activeUsers.Count(u => u.Role == "Patient") + activePatients.Count(),
                    TotalAppointments = appointments.Count(),
                    TotalSessions = sessions.Count(),
                    ActiveModels = 0,
                    AvgResponseTimeMs = 0,
                    HighUrgencyToday = sessions.Count(s => s.UrgencyLevel == "HIGH" && s.CreatedAt.Date == today),
                    SessionsToday = sessions.Count(s => s.CreatedAt.Date == today),
                    SessionsThisWeek = sessions.Count(s => s.CreatedAt.Date >= weekAgo),
                    UrgencyDistribution = urgencyDistribution,
                    SessionsPerDay = sessionsPerDay,
                    UserGrowth = userGrowth
                };
            }
            catch (Exception ex)
            {
                // Log and return safe defaults
                Console.WriteLine($"AdminService.GetSystemStatsAsync error: {ex.Message}");
                return new SystemStatsDto
                {
                    TotalUsers = 0,
                    TotalDoctors = 0,
                    TotalPatients = 0,
                    TotalAppointments = 0,
                    TotalSessions = 0,
                    ActiveModels = 0,
                    AvgResponseTimeMs = 0,
                    HighUrgencyToday = 0,
                    SessionsToday = 0,
                    SessionsThisWeek = 0,
                    UrgencyDistribution = new Dictionary<string, int>
                    {
                        { "LOW", 0 }, 
                        { "MEDIUM", 0 }, 
                        { "HIGH", 0 }, 
                        { "EMERGENCY", 0 }
                    },
                    SessionsPerDay = new List<DateCountDto>(),
                    UserGrowth = new List<DateCountDto>()
                };
            }
        }

        public async Task<PagedResult<UserManagementDto>> GetUsersAsync(int page, int pageSize, string? search, string? role)
        {
            // Get Users (Admin, Doctor roles)
            var allUsers = await _unitOfWork.Repository<User>().GetAllAsync();
            var userQuery = allUsers.Where(u => !u.IsDeleted);

            // Get Patients (registered from mobile app)
            var allPatients = await _unitOfWork.Patients.GetAllAsync();
            var patientQuery = allPatients.Where(p => p.IsActive);

            // Convert Patients to UserManagementDto format
            var patientUsers = patientQuery.Select(p => new UserManagementDto
            {
                Id = p.Id,
                Name = p.FullName,
                Email = p.Email,
                Role = "Patient",
                IsActive = p.IsActive,
                CreatedAt = p.CreatedAt
            });

            // Convert Users to UserManagementDto
            var regularUsers = userQuery.Select(u => new UserManagementDto
            {
                Id = u.Id,
                Name = u.FullName,
                Email = u.Email,
                Role = u.Role,
                IsActive = u.IsActive,
                CreatedAt = u.CreatedAt
            });

            // Combine both lists
            var combined = regularUsers.Concat(patientUsers).ToList();

            // Apply role filter if specified
            if (!string.IsNullOrEmpty(role))
                combined = combined.Where(u => u.Role.Equals(role, StringComparison.OrdinalIgnoreCase)).ToList();

            // Apply search filter if specified
            if (!string.IsNullOrEmpty(search))
                combined = combined.Where(u => u.Name.Contains(search, StringComparison.OrdinalIgnoreCase) ||
                                               u.Email.Contains(search, StringComparison.OrdinalIgnoreCase)).ToList();

            var total = combined.Count;
            var items = combined.Skip((page - 1) * pageSize).Take(pageSize).ToList();

            return new PagedResult<UserManagementDto>
            {
                Items = items,
                Total = total
            };
        }

        public async Task<UserManagementDto> CreateUserAsync(CreateUserRequest request)
        {
            var user = new User
            {
                FullName = request.FullName,
                Email = request.Email,
                PasswordHash = BCrypt.Net.BCrypt.HashPassword(request.Password),
                Role = request.Role,
                PhoneNumber = request.PhoneNumber,
                IsActive = true,
                IsDeleted = false,
                CreatedAt = DateTime.UtcNow
            };

            await _unitOfWork.Repository<User>().AddAsync(user);
            await _unitOfWork.SaveChangesAsync();

            // If Doctor, create Doctor profile
            if (request.Role.Equals("Doctor", StringComparison.OrdinalIgnoreCase))
            {
                var specialtyRepo = _unitOfWork.Repository<Specialty>();
                Specialty? specialty = null;
                var requestedSpecialtyName = string.IsNullOrWhiteSpace(request.SpecialityName)
                    ? "General Practice"
                    : request.SpecialityName.Trim();

                try
                {
                    var specialties = await specialtyRepo.GetAllAsync();
                    specialty = specialties.FirstOrDefault(s =>
                        s.Name.Equals(requestedSpecialtyName, StringComparison.OrdinalIgnoreCase));
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Specialty lookup failed: {ex.Message}");
                }

                if (specialty == null)
                {
                    specialty = new Specialty
                    {
                        Name = requestedSpecialtyName,
                        NameAr = string.IsNullOrWhiteSpace(request.SpecialityNameAr)
                            ? requestedSpecialtyName
                            : request.SpecialityNameAr.Trim()
                    };
                    await specialtyRepo.AddAsync(specialty);
                    await _unitOfWork.SaveChangesAsync();
                }

                var doctor = new Doctor
                {
                    Name = request.FullName,
                    Location = string.Empty,
                    Experience = request.YearsExperience ?? 0,
                    ConsultationFee = request.ConsultationFee ?? 0,
                    Bio = request.Bio ?? string.Empty,
                    IsAvailable = true,
                    Rating = 0,
                    ReviewCount = 0,
                    SpecialtyId = specialty.Id,
                    UserId = user.Id
                };

                await _unitOfWork.Repository<Doctor>().AddAsync(doctor);
                await _unitOfWork.SaveChangesAsync();

                await _notificationService.NotifyDoctorCreated(
                    request.FullName,
                    request.SpecialityName ?? "General");
            }

            return _mapper.Map<UserManagementDto>(user);
        }

        public async Task<bool> ToggleUserStatusAsync(int userId)
        {
            var user = await _unitOfWork.Repository<User>().GetByIdAsync(userId);
            if (user != null)
            {
                return user.IsActive
                    ? await DeactivateUserAsync(userId, user.Role)
                    : await ActivateUserAsync(userId, user.Role);
            }

            var patient = await _unitOfWork.Patients.GetByIdAsync(userId);
            if (patient == null) return false;

            return patient.IsActive
                ? await DeactivateUserAsync(userId, "Patient")
                : await ActivateUserAsync(userId, "Patient");
        }

        public async Task<bool> DeactivateUserAsync(int userId, string role)
        {
            if (string.Equals(role, "Patient", StringComparison.OrdinalIgnoreCase))
            {
                var patient = await _unitOfWork.Patients.GetByIdAsync(userId);
                if (patient == null) return false;

                patient.IsActive = false;
                _unitOfWork.Patients.Update(patient);
                await _unitOfWork.SaveChangesAsync();
                return true;
            }

            var user = await _unitOfWork.Repository<User>().GetByIdAsync(userId);
            if (user == null) return false;

            user.IsActive = false;
            user.UpdatedAt = DateTime.UtcNow;

            if (string.Equals(user.Role, "Doctor", StringComparison.OrdinalIgnoreCase))
            {
                await SetDoctorMobileAvailabilityAsync(user.Id, false);
            }

            await _unitOfWork.SaveChangesAsync();
            return true;
        }

        public async Task<bool> ActivateUserAsync(int userId, string role)
        {
            if (string.Equals(role, "Patient", StringComparison.OrdinalIgnoreCase))
            {
                var patient = await _unitOfWork.Patients.GetByIdAsync(userId);
                if (patient == null) return false;

                patient.IsActive = true;
                _unitOfWork.Patients.Update(patient);
                await _unitOfWork.SaveChangesAsync();
                return true;
            }

            var user = await _unitOfWork.Repository<User>().GetByIdAsync(userId);
            if (user == null) return false;

            user.IsActive = true;
            user.UpdatedAt = DateTime.UtcNow;
            await _unitOfWork.SaveChangesAsync();
            return true;
        }

        public async Task<bool> DeleteUserAsync(int id, string role)
        {
            if (string.Equals(role, "Patient", StringComparison.OrdinalIgnoreCase))
            {
                var patient = await _unitOfWork.Patients.GetByIdAsync(id);
                if (patient == null) return false;

                var existingTombstone = (await _unitOfWork.Repository<User>()
                    .FindAsync(u => u.Email.ToLower() == patient.Email.ToLower()))
                    .FirstOrDefault();

                if (existingTombstone == null)
                {
                    var tombstone = new User
                    {
                        FullName = patient.FullName,
                        Email = patient.Email,
                        PasswordHash = patient.PasswordHash,
                        Role = "Patient",
                        PhoneNumber = patient.PhoneNumber,
                        IsActive = false,
                        IsDeleted = true,
                        CreatedAt = patient.CreatedAt,
                        UpdatedAt = DateTime.UtcNow
                    };

                    await _unitOfWork.Repository<User>().AddAsync(tombstone);
                }

                _unitOfWork.Patients.Delete(patient);
                await _unitOfWork.SaveChangesAsync();
                return true;
            }

            var user = await _unitOfWork.Repository<User>().GetByIdAsync(id);
            if (user == null) return false;

            user.IsDeleted = true;
            user.IsActive = false;
            user.UpdatedAt = DateTime.UtcNow;

            if (string.Equals(user.Role, "Doctor", StringComparison.OrdinalIgnoreCase))
            {
                await SetDoctorMobileAvailabilityAsync(user.Id, false);
            }

            await _unitOfWork.SaveChangesAsync();
            return true;
        }

        public Task<IEnumerable<ModelVersionDto>> ListModelVersionsAsync()
        {
            return Task.FromResult(Enumerable.Empty<ModelVersionDto>());
        }

        public Task ReloadAiModelAsync(string agentName)
        {
            return Task.CompletedTask;
        }

        private async Task SetDoctorMobileAvailabilityAsync(int userId, bool isMobileEnabled)
        {
            var doctor = (await _unitOfWork.Repository<Doctor>()
                .FindAsync(d => d.UserId == userId))
                .FirstOrDefault();

            if (doctor == null) return;

            doctor.IsAvailable = isMobileEnabled;
            _unitOfWork.Repository<Doctor>().Update(doctor);
        }
    }
}
