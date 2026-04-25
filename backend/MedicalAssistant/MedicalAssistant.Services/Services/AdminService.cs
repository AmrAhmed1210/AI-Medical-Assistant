using AutoMapper;
using MedicalAssistant.Shared.DTOs.DoctorDTOs;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.UserModule;
using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Domain.Entities.AppointmentsModule;
using MedicalAssistant.Domain.Entities.ConsultationsModule;
using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Domain.Entities.SessionsModule;
using MedicalAssistant.Domain.Entities.ReviewsModule;
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
                PhotoUrl = p.ImageUrl,
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
                PhotoUrl = u.PhotoUrl,
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
                PhotoUrl = request.PhotoUrl,
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

                var doctorRepo = _unitOfWork.Repository<Doctor>();
                var existingDoctor = (await doctorRepo.FindAsync(d => d.UserId == user.Id)).FirstOrDefault();
                if (existingDoctor == null)
                {
                    var doctor = new Doctor
                    {
                        Name = request.FullName,
                        Location = string.Empty,
                        Experience = request.YearsExperience ?? 0,
                        ConsultationFee = request.ConsultationFee ?? 0,
                        Bio = request.Bio ?? string.Empty,
                        IsAvailable = true,
                        IsScheduleVisible = true,
                        Rating = 0,
                        ReviewCount = 0,
                        SpecialtyId = specialty.Id,
                        ImageUrl = request.PhotoUrl,
                        UserId = user.Id
                    };

                    await doctorRepo.AddAsync(doctor);
                    await _unitOfWork.SaveChangesAsync();
                }

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
            await _unitOfWork.BeginTransactionAsync();
            try
            {
                if (string.Equals(role, "Patient", StringComparison.OrdinalIgnoreCase))
                {
                    var patient = await _unitOfWork.Patients.GetByIdAsync(id);
                    if (patient == null) { await _unitOfWork.RollbackTransactionAsync(); return false; }

                    // 1. Delete Follows
                    var follows = (await _unitOfWork.Repository<FollowedDoctor>().FindAsync(f => f.PatientId == patient.Id)).ToList();
                    if (follows.Any()) _unitOfWork.Repository<FollowedDoctor>().DeleteRange(follows);

                    // 2. Delete Reviews
                    var reviews = (await _unitOfWork.Repository<Review>().FindAsync(r => r.PatientId == patient.Id)).ToList();
                    if (reviews.Any()) _unitOfWork.Repository<Review>().DeleteRange(reviews);

                    // 3. Delete Appointments
                    var appointments = (await _unitOfWork.Repository<Appointment>().FindAsync(a => a.PatientId == patient.Id)).ToList();
                    if (appointments.Any()) _unitOfWork.Repository<Appointment>().DeleteRange(appointments);

                    // 4. Delete Consultations
                    var consultations = (await _unitOfWork.Repository<Consultation>().FindAsync(c => c.PatientId == patient.Id)).ToList();
                    if (consultations.Any()) _unitOfWork.Repository<Consultation>().DeleteRange(consultations);

                    // 5. Delete Sessions + Messages
                    var sessions = (await _unitOfWork.Repository<Session>().FindAsync(s => s.UserId == patient.Id)).ToList();
                    if (sessions.Any())
                    {
                        foreach (var s in sessions)
                        {
                            var messages = (await _unitOfWork.Repository<Message>().FindAsync(m => m.SessionId == s.Id)).ToList();
                            if (messages.Any()) _unitOfWork.Repository<Message>().DeleteRange(messages);
                        }
                        _unitOfWork.Repository<Session>().DeleteRange(sessions);
                    }

                    _unitOfWork.Patients.Delete(patient);
                    await _unitOfWork.CommitTransactionAsync();
                    return true;
                }

                var user = await _unitOfWork.Repository<User>().GetByIdAsync(id);
                if (user == null) { await _unitOfWork.RollbackTransactionAsync(); return false; }

                if (string.Equals(user.Role, "Doctor", StringComparison.OrdinalIgnoreCase))
                {
                    var doctor = (await _unitOfWork.Repository<Doctor>().FindAsync(d => d.UserId == user.Id)).FirstOrDefault();
                    if (doctor != null)
                    {
                        // 1. Delete Availabilities
                        var availabilities = (await _unitOfWork.Repository<DoctorAvailability>().FindAsync(a => a.DoctorId == doctor.Id)).ToList();
                        if (availabilities.Any()) _unitOfWork.Repository<DoctorAvailability>().DeleteRange(availabilities);

                        // 2. Delete Follows
                        var follows = (await _unitOfWork.Repository<FollowedDoctor>().FindAsync(f => f.DoctorId == doctor.Id)).ToList();
                        if (follows.Any()) _unitOfWork.Repository<FollowedDoctor>().DeleteRange(follows);

                        // 3. Delete Reviews
                        var reviews = (await _unitOfWork.Repository<Review>().FindAsync(r => r.DoctorId == doctor.Id)).ToList();
                        if (reviews.Any()) _unitOfWork.Repository<Review>().DeleteRange(reviews);

                        // 4. Delete Appointments
                        var appointments = (await _unitOfWork.Repository<Appointment>().FindAsync(a => a.DoctorId == doctor.Id)).ToList();
                        if (appointments.Any()) _unitOfWork.Repository<Appointment>().DeleteRange(appointments);

                        // 5. Delete Consultations
                        var consultations = (await _unitOfWork.Repository<Consultation>().FindAsync(c => c.DoctorId == doctor.Id)).ToList();
                        if (consultations.Any()) _unitOfWork.Repository<Consultation>().DeleteRange(consultations);

                        _unitOfWork.Repository<Doctor>().Delete(doctor);
                    }
                }

                _unitOfWork.Repository<User>().Delete(user);
                await _unitOfWork.CommitTransactionAsync();
                return true;
            }
            catch (Exception ex)
            {
                await _unitOfWork.RollbackTransactionAsync();
                Console.WriteLine($"Delete failed: {ex.Message}");
                throw;
            }
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

        public async Task<IEnumerable<DoctorApplicationDto>> GetDoctorApplicationsAsync(string? status = null)
        {
            var apps = await _unitOfWork.Repository<DoctorApplication>().GetAllAsync();
            if (!string.IsNullOrEmpty(status))
                apps = apps.Where(a => a.Status.Equals(status, StringComparison.OrdinalIgnoreCase)).ToList();

            var specialtyIds = apps.Select(a => a.SpecialtyId).Distinct().ToList();
            var specialties = await _unitOfWork.Repository<Specialty>().FindAsync(s => specialtyIds.Contains(s.Id));
            var specDict = specialties.ToDictionary(s => s.Id, s => s.Name);

            return apps.Select(a => new DoctorApplicationDto
            {
                Id = a.Id,
                Name = a.Name,
                Email = a.Email,
                Phone = a.Phone,
                SpecialtyId = a.SpecialtyId,
                SpecialtyName = specDict.GetValueOrDefault(a.SpecialtyId, "General"),
                Experience = a.Experience,
                Bio = a.Bio,
                LicenseNumber = a.LicenseNumber,
                Message = a.Message,
                DocumentUrl = a.DocumentUrl,
                PhotoUrl = a.PhotoUrl,
                Status = a.Status,
                SubmittedAt = a.SubmittedAt,
                ProcessedAt = a.ProcessedAt
            }).OrderByDescending(a => a.SubmittedAt).ToList();
        }

        public async Task<bool> ApproveDoctorApplicationAsync(int applicationId)
        {
            var app = await _unitOfWork.Repository<DoctorApplication>().GetByIdAsync(applicationId);
            if (app == null || app.Status != "Pending") return false;

            app.Status = "Approved";
            app.ProcessedAt = DateTime.UtcNow;

            var user = new User
            {
                FullName = app.Name,
                Email = app.Email,
                PasswordHash = BCrypt.Net.BCrypt.HashPassword("Doctor@123!"), // Default password
                Role = "Doctor",
                PhoneNumber = app.Phone,
                PhotoUrl = app.PhotoUrl,
                IsActive = true,
                IsDeleted = false,
                CreatedAt = DateTime.UtcNow
            };

            await _unitOfWork.Repository<User>().AddAsync(user);
            await _unitOfWork.SaveChangesAsync();

            var doctor = new Doctor
            {
                Name = app.Name,
                Location = string.Empty,
                Experience = app.Experience,
                ConsultationFee = 0,
                Bio = app.Bio,
                IsAvailable = true,
                IsScheduleVisible = true,
                Rating = 0,
                ReviewCount = 0,
                SpecialtyId = app.SpecialtyId,
                UserId = user.Id
            };

            await _unitOfWork.Repository<Doctor>().AddAsync(doctor);
            await _unitOfWork.SaveChangesAsync();

            // Notify admin group that application was approved
            await _notificationService.NotifyNewUserRegistered(user.Id, user.FullName, user.Email, "Doctor");
            return true;
        }

        public async Task<bool> RejectDoctorApplicationAsync(int applicationId, string? reason = null)
        {
            var app = await _unitOfWork.Repository<DoctorApplication>().GetByIdAsync(applicationId);
            if (app == null || app.Status != "Pending") return false;

            app.Status = "Rejected";
            app.ProcessedAt = DateTime.UtcNow;
            // Optionally store reason in Message field for audit
            if (!string.IsNullOrWhiteSpace(reason))
                app.Message = $"[REJECTION REASON]: {reason}";

            await _unitOfWork.SaveChangesAsync();
            return true;
        }
    }
}
