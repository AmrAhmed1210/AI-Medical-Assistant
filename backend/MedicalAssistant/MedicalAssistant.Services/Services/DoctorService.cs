using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.AppointmentsModule;
using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Domain.Entities.SessionsModule;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.AppointmentsDTOs;
using MedicalAssistant.Shared.DTOs.DoctorDTOs;
using System.Globalization;
using System.Linq.Expressions;

namespace MedicalAssistant.Application.Services;

public class DoctorService : IDoctorService
{
    private readonly IUnitOfWork _unitOfWork;
    private readonly INotificationService _notificationService;

    public DoctorService(IUnitOfWork unitOfWork, AutoMapper.IMapper mapper, INotificationService notificationService)
    {
        _unitOfWork = unitOfWork;
        _notificationService = notificationService;
    }

    public async Task<IReadOnlyList<DoctorDTO>> GetAllDoctorsAsync()
    {
        var doctors = (await _unitOfWork.Repository<Doctor>().GetAllAsync(d => d.Specialty, d => d.User!)).ToList();
        var filteredDoctors = FilterActiveDoctors(doctors);
        return await MapDoctorsAsync(filteredDoctors);
    }

    public async Task<DoctorDetailsDTO?> GetDoctorByIdAsync(int id)
    {
        var doctor = await _unitOfWork.Repository<Doctor>().GetByIdAsync(id, d => d.Specialty, d => d.User!);
        if (doctor is null || !IsDoctorUserActive(doctor))
        {
            return null;
        }

        var schedule = await BuildScheduleDtoAsync(doctor);
        return MapDoctorDetailsDto(doctor, schedule);
    }

    public async Task<IReadOnlyList<DoctorDTO>> GetAvailableDoctorsAsync()
    {
        var doctors = (await _unitOfWork.Doctors.GetAvailableDoctorsAsync()).ToList();
        var users = (await _unitOfWork.Repository<MedicalAssistant.Domain.Entities.UserModule.User>()
            .FindAsync(u => doctors.Where(d => d.UserId.HasValue).Select(d => d.UserId!.Value).Contains(u.Id)))
            .ToDictionary(u => u.Id);

        foreach (var doctor in doctors.Where(d => d.UserId.HasValue && users.ContainsKey(d.UserId.Value)))
        {
            doctor.User = users[doctor.UserId!.Value];
        }

        var filteredDoctors = FilterActiveDoctors(doctors);
        return await MapDoctorsAsync(filteredDoctors.Where(d => d.IsAvailable));
    }

    public async Task<IReadOnlyList<DoctorDTO>> GetDoctorsBySpecialtyAsync(int specialtyId)
    {
        var doctors = (await _unitOfWork.Doctors.GetBySpecialtyAsync(specialtyId)).ToList();
        var users = (await _unitOfWork.Repository<MedicalAssistant.Domain.Entities.UserModule.User>()
            .FindAsync(u => doctors.Where(d => d.UserId.HasValue).Select(d => d.UserId!.Value).Contains(u.Id)))
            .ToDictionary(u => u.Id);

        foreach (var doctor in doctors.Where(d => d.UserId.HasValue && users.ContainsKey(d.UserId.Value)))
        {
            doctor.User = users[doctor.UserId!.Value];
        }

        var filteredDoctors = FilterActiveDoctors(doctors);
        return await MapDoctorsAsync(filteredDoctors);
    }

    public async Task<IReadOnlyList<DoctorDTO>> SearchDoctorsAsync(string name)
    {
        var doctors = (await _unitOfWork.Doctors.SearchByNameAsync(name)).ToList();
        var users = (await _unitOfWork.Repository<MedicalAssistant.Domain.Entities.UserModule.User>()
            .FindAsync(u => doctors.Where(d => d.UserId.HasValue).Select(d => d.UserId!.Value).Contains(u.Id)))
            .ToDictionary(u => u.Id);

        foreach (var doctor in doctors.Where(d => d.UserId.HasValue && users.ContainsKey(d.UserId.Value)))
        {
            doctor.User = users[doctor.UserId!.Value];
        }

        var filteredDoctors = FilterActiveDoctors(doctors);
        return await MapDoctorsAsync(filteredDoctors);
    }

    public async Task<IReadOnlyList<DoctorDTO>> GetTopRatedDoctorsAsync(int count)
    {
        var doctors = (await _unitOfWork.Doctors.GetTopRatedDoctorsAsync(count)).ToList();
        var users = (await _unitOfWork.Repository<MedicalAssistant.Domain.Entities.UserModule.User>()
            .FindAsync(u => doctors.Where(d => d.UserId.HasValue).Select(d => d.UserId!.Value).Contains(u.Id)))
            .ToDictionary(u => u.Id);

        foreach (var doctor in doctors.Where(d => d.UserId.HasValue && users.ContainsKey(d.UserId.Value)))
        {
            doctor.User = users[doctor.UserId!.Value];
        }

        var filteredDoctors = FilterActiveDoctors(doctors);
        return await MapDoctorsAsync(filteredDoctors);
    }

    public async Task<(IReadOnlyList<DoctorDTO> Items, int TotalCount)> GetPaginatedDoctorsAsync(int pageNumber, int pageSize)
    {
        var (items, _) = await _unitOfWork.Doctors.GetPaginatedAsync(pageNumber, pageSize);
        var doctors = items.ToList();
        var users = (await _unitOfWork.Repository<MedicalAssistant.Domain.Entities.UserModule.User>()
            .FindAsync(u => doctors.Where(d => d.UserId.HasValue).Select(d => d.UserId!.Value).Contains(u.Id)))
            .ToDictionary(u => u.Id);

        foreach (var doctor in doctors.Where(d => d.UserId.HasValue && users.ContainsKey(d.UserId.Value)))
        {
            doctor.User = users[doctor.UserId!.Value];
        }

        var filteredDoctors = FilterActiveDoctors(doctors);
        var mappedItems = await MapDoctorsAsync(filteredDoctors);
        return (mappedItems, mappedItems.Count);
    }

    public async Task<DoctorDashboardDto> GetDashboardAsync(int doctorId)
    {
        var doctor = await GetDoctorByUserIdAsync(doctorId);
        if (doctor is null)
        {
            return new DoctorDashboardDto();
        }

        var appointments = (await _unitOfWork.Repository<Appointment>()
            .FindAsync(a => a.DoctorId == doctor.Id, a => a.Patient))
            .ToList();

        var today = DateTime.UtcNow.Date;
        var weekStart = today.AddDays(-7);

        var appointmentTimes = appointments
            .Select(a => new { Appointment = a, ParsedDate = ParseAppointmentDateTime(a.Date, a.Time) })
            .ToList();

        var todayAppointments = appointmentTimes
            .Where(x => x.ParsedDate.HasValue && x.ParsedDate.Value.Date == today)
            .Select(x => x.Appointment)
            .ToList();

        var weekAppointments = appointmentTimes.Count(x => x.ParsedDate.HasValue && x.ParsedDate.Value.Date >= weekStart);
        var pendingAppointments = appointments.Count(a => string.Equals(a.Status, "Pending", StringComparison.OrdinalIgnoreCase));
        var totalPatients = appointments.Select(a => a.PatientId).Distinct().Count();

        var sessions = (await _unitOfWork.Repository<Session>().GetAllAsync())
            .Where(s => s.CreatedAt.Date >= weekStart)
            .GroupBy(s => s.CreatedAt.DayOfWeek)
            .ToDictionary(g => g.Key, g => g.Count());

        var orderedDays = new[]
        {
            DayOfWeek.Sunday, DayOfWeek.Monday, DayOfWeek.Tuesday, DayOfWeek.Wednesday,
            DayOfWeek.Thursday, DayOfWeek.Friday, DayOfWeek.Saturday
        };

        var weeklyChart = orderedDays
            .Select(day => new ChartDataDto
            {
                Day = day.ToString()[..3],
                Count = sessions.GetValueOrDefault(day, 0)
            })
            .ToList();

        return new DoctorDashboardDto
        {
            TodayAppointments = todayAppointments.Count,
            PendingAppointments = pendingAppointments,
            TotalPatients = totalPatients,
            WeekAppointments = weekAppointments,
            TodayAppointmentsList = todayAppointments.Select(MapAppointment).ToList(),
            WeeklySessionsChart = weeklyChart,
            RecentReports = new List<AIReportDto>()
        };
    }

    public async Task<DoctorDetailDto?> GetProfileAsync(int doctorId)
    {
        var doctor = await GetDoctorByUserIdAsync(doctorId, includeUser: true, includeSpecialty: true);
        if (doctor is null)
        {
            return null;
        }

        return new DoctorDetailDto
        {
            Id = doctor.Id,
            UserId = doctor.UserId,
            FullName = doctor.Name,
            Email = doctor.User?.Email ?? string.Empty,
            Specialty = doctor.Specialty?.Name ?? string.Empty,
            SpecialityNameAr = doctor.Specialty?.NameAr,
            Bio = doctor.Bio,
            PhotoUrl = doctor.ImageUrl,
            ConsultFee = doctor.ConsultationFee,
            YearsExperience = doctor.Experience,
            IsAvailable = doctor.IsAvailable,
            CreatedAt = DateTime.UtcNow,
            UpdatedAt = null
        };
    }

    public async Task UpdateProfileAsync(int doctorId, UpdateDoctorProfileRequest request)
    {
        var doctor = await GetDoctorByUserIdAsync(doctorId, includeUser: true);
        if (doctor is null)
        {
            return;
        }

        doctor.Name = string.IsNullOrWhiteSpace(request.FullName) ? doctor.Name : request.FullName.Trim();
        doctor.Bio = request.Bio ?? string.Empty;
        doctor.Experience = request.YearsExperience;
        doctor.ConsultationFee = request.ConsultationFee;
        doctor.IsAvailable = request.IsAvailable;

        if (doctor.User is not null && !string.IsNullOrWhiteSpace(request.FullName))
        {
            doctor.User.FullName = request.FullName.Trim();
            doctor.User.UpdatedAt = DateTime.UtcNow;
        }

        _unitOfWork.Repository<Doctor>().Update(doctor);
        await _unitOfWork.SaveChangesAsync();
        await _notificationService.NotifyProfileUpdated(doctor.Id, doctor.Name);
    }

    public async Task<IEnumerable<AppointmentDto>> GetAppointmentsAsync(int doctorId, string? status)
    {
        var doctor = await GetDoctorByUserIdAsync(doctorId);
        if (doctor is null)
        {
            return Enumerable.Empty<AppointmentDto>();
        }

        var appointments = (await _unitOfWork.Repository<Appointment>()
            .FindAsync(a => a.DoctorId == doctor.Id, a => a.Patient, a => a.Doctor))
            .ToList();

        if (!string.IsNullOrWhiteSpace(status))
        {
            appointments = appointments
                .Where(a => string.Equals(a.Status, status, StringComparison.OrdinalIgnoreCase))
                .ToList();
        }

        return appointments.Select(MapAppointment);
    }

    public async Task<IEnumerable<PatientSummaryDto>> GetPatientsAsync(int doctorId, string? search)
    {
        var doctor = await GetDoctorByUserIdAsync(doctorId);
        if (doctor is null)
        {
            return Enumerable.Empty<PatientSummaryDto>();
        }

        var appointments = (await _unitOfWork.Repository<Appointment>()
            .FindAsync(a => a.DoctorId == doctor.Id))
            .ToList();

        var patientIds = appointments.Select(a => a.PatientId).Distinct().ToList();
        if (patientIds.Count == 0)
        {
            return Enumerable.Empty<PatientSummaryDto>();
        }

        var patients = (await _unitOfWork.Repository<Patient>().FindAsync(p => patientIds.Contains(p.Id))).ToList();

        var summaries = patients.Select(patient =>
        {
            var patientAppointments = appointments.Where(a => a.PatientId == patient.Id).ToList();
            return new PatientSummaryDto
            {
                Id = patient.Id,
                FullName = patient.FullName,
                Email = patient.Email,
                PhoneNumber = patient.PhoneNumber,
                DateOfBirth = patient.DateOfBirth,
                Gender = patient.Gender,
                BloodType = patient.BloodType,
                Allergies = patient.MedicalNotes,
                TotalAppointments = patientAppointments.Count,
                LastVisit = patientAppointments.OrderByDescending(a => a.CreatedAt).FirstOrDefault()?.CreatedAt.ToString("O")
            };
        });

        if (!string.IsNullOrWhiteSpace(search))
        {
            summaries = summaries.Where(p =>
                p.FullName.Contains(search, StringComparison.OrdinalIgnoreCase) ||
                p.Email.Contains(search, StringComparison.OrdinalIgnoreCase));
        }

        return summaries.ToList();
    }

    public Task<IEnumerable<AIReportDto>> GetReportsAsync(int doctorId, string? urgency)
    {
        return Task.FromResult<IEnumerable<AIReportDto>>(new List<AIReportDto>());
    }

    public async Task<IEnumerable<AvailabilityDto>> GetAvailabilityAsync(int doctorId)
    {
        var schedule = await GetMyScheduleAsync(doctorId);
        return schedule?.Days ?? Enumerable.Empty<AvailabilityDto>();
    }

    public async Task<IEnumerable<AvailabilityDto>> GetPublicAvailabilityAsync(int doctorId)
    {
        var schedule = await GetScheduleAsync(doctorId);
        return schedule?.Days ?? Enumerable.Empty<AvailabilityDto>();
    }

    public async Task UpdateAvailabilityAsync(int doctorId, List<AvailabilityDto> data)
    {
        var doctor = await GetDoctorByUserIdAsync(doctorId);
        if (doctor is null)
        {
            return;
        }

        await UpdateScheduleAsync(doctorId, new UpdateDoctorScheduleRequest
        {
            IsMobileEnabled = doctor.IsAvailable,
            Days = data
        });
    }

    public async Task<DoctorScheduleDto?> GetScheduleAsync(int doctorId)
    {
        var doctor = await _unitOfWork.Repository<Doctor>().GetByIdAsync(doctorId, d => d.Specialty, d => d.User!);
        if (doctor is null || !IsDoctorUserActive(doctor))
        {
            return null;
        }

        return await BuildScheduleDtoAsync(doctor);
    }

    public async Task<DoctorScheduleDto?> GetMyScheduleAsync(int doctorUserId)
    {
        var doctor = await GetDoctorByUserIdAsync(doctorUserId, includeSpecialty: true, includeUser: true);
        if (doctor is null)
        {
            return null;
        }

        return await BuildScheduleDtoAsync(doctor);
    }

    public async Task UpdateScheduleAsync(int doctorUserId, UpdateDoctorScheduleRequest request)
    {
        var doctor = await GetDoctorByUserIdAsync(doctorUserId);
        if (doctor is null)
        {
            return;
        }

        doctor.IsAvailable = request.IsMobileEnabled;
        _unitOfWork.Repository<Doctor>().Update(doctor);

        var availabilityRepo = _unitOfWork.Repository<DoctorAvailability>();
        var existing = (await availabilityRepo.FindAsync(a => a.DoctorId == doctor.Id)).ToList();
        if (existing.Count > 0)
        {
            availabilityRepo.DeleteRange(existing);
        }

        var rowsToInsert = request.Days
            .GroupBy(day => day.DayOfWeek)
            .Select(group => group.Last())
            .OrderBy(day => day.DayOfWeek)
            .Select(item => new DoctorAvailability
            {
                DoctorId = doctor.Id,
                DayOfWeek = item.DayOfWeek,
                StartTime = ParseTime(item.StartTime, new TimeSpan(9, 0, 0)),
                EndTime = ParseTime(item.EndTime, new TimeSpan(17, 0, 0)),
                IsAvailable = item.IsAvailable,
                SlotDurationMinutes = item.SlotDurationMinutes > 0 ? item.SlotDurationMinutes : 30
            })
            .ToList();

        if (rowsToInsert.Count > 0)
        {
            await availabilityRepo.AddRangeAsync(rowsToInsert);
        }

        await _unitOfWork.SaveChangesAsync();

        // Notify subscribers via SignalR targeted group
        await _notificationService.NotifyScheduleReady(doctor.Id, doctor.Name);
    }

    public async Task<IEnumerable<MedicalAssistant.Shared.DTOs.ReviewDTOs.ReviewDto>> GetMyReviewsAsync(int userId)
    {
        var doctor = (await _unitOfWork.Repository<Doctor>().FindAsync(d => d.UserId == userId)).FirstOrDefault();
        if (doctor == null) return Enumerable.Empty<MedicalAssistant.Shared.DTOs.ReviewDTOs.ReviewDto>();

        var reviews = await _unitOfWork.Repository<MedicalAssistant.Domain.Entities.ReviewsModule.Review>()
            .FindAsync(r => r.DoctorId == doctor.Id);

        return reviews.Select(r => new MedicalAssistant.Shared.DTOs.ReviewDTOs.ReviewDto
        {
            Id = r.Id.ToString(),
            Author = r.Author,
            PatientName = r.PatientName ?? "Anonymous",
            Rating = r.Rating,
            Comment = r.Comment,
            CreatedAt = r.CreatedAt
        });
    }

    public async Task<IEnumerable<SpecialtyDto>> GetSpecialtiesAsync()
    {
        var specialties = await _unitOfWork.Repository<Specialty>().GetAllAsync();
        return specialties
            .OrderBy(s => s.Name)
            .Select(s => new SpecialtyDto
            {
                Id = s.Id,
                Name = s.Name,
                NameAr = s.NameAr
            })
            .ToList();
    }

    // Legacy methods kept for existing callers.
    public Task<DoctorDashboardDto> GetDashboardStatsAsync(int doctorId) => GetDashboardAsync(doctorId);

    public async Task<DoctorDetailsDTO> GetDoctorProfileAsync(int doctorId)
    {
        var doctor = await GetDoctorByUserIdAsync(doctorId, includeSpecialty: true, includeUser: true);
        if (doctor is null)
        {
            return new DoctorDetailsDTO();
        }

        var schedule = await BuildScheduleDtoAsync(doctor);
        return MapDoctorDetailsDto(doctor, schedule);
    }

    public Task<IEnumerable<AppointmentDto>> GetDoctorAppointmentsAsync(int doctorId) => GetAppointmentsAsync(doctorId, null);

    public async Task<IEnumerable<MedicalAssistant.Shared.DTOs.PatientDTOs.PatientDto>> GetDoctorPatientsAsync(int doctorId)
    {
        var summaries = await GetPatientsAsync(doctorId, null);
        return summaries.Select(p => new MedicalAssistant.Shared.DTOs.PatientDTOs.PatientDto
        {
            Id = p.Id,
            FullName = p.FullName,
            Email = p.Email,
            PhoneNumber = p.PhoneNumber ?? string.Empty,
            DateOfBirth = p.DateOfBirth ?? DateTime.MinValue,
            Gender = p.Gender ?? string.Empty,
            BloodType = p.BloodType,
            MedicalNotes = p.Allergies
        });
    }

    public Task<IEnumerable<AIReportDto>> GetDoctorReportsAsync(int doctorId) => GetReportsAsync(doctorId, null);

    public Task<IEnumerable<AvailabilityDto>> GetDoctorAvailabilityAsync(int doctorId) => GetAvailabilityAsync(doctorId);

    public async Task UpdateScheduleVisibilityAsync(int doctorUserId, bool isVisible)
    {
        var doctor = await GetDoctorByUserIdAsync(doctorUserId);
        if (doctor is null)
        {
            return;
        }

        doctor.IsScheduleVisible = isVisible;
        await _unitOfWork.SaveChangesAsync();
    }

    public async Task<bool> SelfDeactivateAsync(int doctorUserId)
    {
        var doctor = await GetDoctorByUserIdAsync(doctorUserId, includeUser: true);
        if (doctor is null || doctor.User is null)
        {
            return false;
        }

        doctor.User.IsActive = false;
        doctor.User.UpdatedAt = DateTime.UtcNow;
        doctor.IsAvailable = false;

        await _unitOfWork.SaveChangesAsync();
        return true;
    }

    private async Task<IReadOnlyList<DoctorDTO>> MapDoctorsAsync(IEnumerable<Doctor> doctors)
    {
        var doctorList = doctors.ToList();
        if (doctorList.Count == 0)
        {
            return Array.Empty<DoctorDTO>();
        }

        var scheduleMap = await GetScheduleMapAsync(doctorList.Select(d => d.Id));
        return doctorList
            .Select(doctor => MapDoctorDto(doctor, scheduleMap.GetValueOrDefault(doctor.Id)))
            .ToList();
    }

    private DoctorDTO MapDoctorDto(Doctor doctor, bool hasSchedule)
    {
        return new DoctorDTO
        {
            Id = doctor.Id,
            Name = doctor.Name,
            Specialty = doctor.Specialty?.Name ?? string.Empty,
            Rating = doctor.Rating,
            ReviewCount = doctor.ReviewCount,
            Location = doctor.Location,
            ConsultationFee = doctor.ConsultationFee,
            IsAvailable = doctor.IsAvailable && hasSchedule,
            ImageUrl = doctor.ImageUrl ?? string.Empty,
            YearsExperience = doctor.Experience,
            IsProfileComplete = IsProfileComplete(doctor),
            IsMobileEnabled = doctor.IsAvailable,
            HasSchedule = hasSchedule,
            IsScheduleVisible = doctor.IsScheduleVisible
        };
    }

    private DoctorDetailsDTO MapDoctorDetailsDto(Doctor doctor, DoctorScheduleDto schedule)
    {
        return new DoctorDetailsDTO
        {
            Id = doctor.Id,
            Name = doctor.Name,
            Specialty = doctor.Specialty?.Name ?? string.Empty,
            Rating = doctor.Rating,
            ReviewCount = doctor.ReviewCount,
            Location = doctor.Location,
            ConsultationFee = doctor.ConsultationFee,
            IsAvailable = doctor.IsAvailable && schedule.HasSchedule,
            ImageUrl = doctor.ImageUrl ?? string.Empty,
            YearsExperience = doctor.Experience,
            IsProfileComplete = schedule.IsProfileComplete,
            IsMobileEnabled = doctor.IsAvailable,
            HasSchedule = schedule.HasSchedule,
            IsScheduleVisible = doctor.IsScheduleVisible,
            Experience = doctor.Experience,
            Bio = doctor.Bio ?? string.Empty,
            Schedule = schedule
        };
    }

    private async Task<DoctorScheduleDto> BuildScheduleDtoAsync(Doctor doctor)
    {
        var rows = (await _unitOfWork.Repository<DoctorAvailability>().FindAsync(a => a.DoctorId == doctor.Id))
            .OrderBy(a => a.DayOfWeek)
            .ToList();

        // Fetch booked slots (Pending or Confirmed appointments) for the next 30 days
        var startDate = DateTime.UtcNow.Date;
        var endDate = startDate.AddDays(30);
        
        var appointments = await _unitOfWork.Repository<Appointment>()
            .FindAsync(a => a.DoctorId == doctor.Id && 
                            (a.Status == "Pending" || a.Status == "Confirmed"));
        
        var bookedSlots = appointments
            .Select(a => new BookedSlotDto 
            { 
                Date = a.Date, 
                Time = a.Time 
            })
            .ToList();

        // If schedule is not visible to patients, return empty schedule but with booked slots
        if (!doctor.IsScheduleVisible)
        {
            return new DoctorScheduleDto
            {
                DoctorId = doctor.Id,
                DoctorName = doctor.Name,
                IsMobileEnabled = doctor.IsAvailable,
                IsProfileComplete = IsProfileComplete(doctor),
                HasSchedule = false,
                Days = new List<AvailabilityDto>(),
                BookedSlots = bookedSlots
            };
        }

        var rowsByDay = rows.ToDictionary(row => row.DayOfWeek, row => row);
        var days = Enumerable.Range(0, 7)
            .Select(index =>
            {
                var dayOfWeek = (byte)index;
                if (rowsByDay.TryGetValue(dayOfWeek, out var row))
                {
                    return MapAvailability(row);
                }

                return new AvailabilityDto
                {
                    DayOfWeek = dayOfWeek,
                    DayName = GetDayName(dayOfWeek),
                    StartTime = "09:00",
                    EndTime = "17:00",
                    IsAvailable = false,
                    SlotDurationMinutes = 30
                };
            })
            .ToList();

        return new DoctorScheduleDto
        {
            DoctorId = doctor.Id,
            DoctorName = doctor.Name,
            IsMobileEnabled = doctor.IsAvailable,
            IsProfileComplete = IsProfileComplete(doctor),
            HasSchedule = rows.Any(row => row.IsAvailable),
            Days = days,
            BookedSlots = bookedSlots
        };
    }

    private static AvailabilityDto MapAvailability(DoctorAvailability row)
    {
        return new AvailabilityDto
        {
            DayOfWeek = row.DayOfWeek,
            DayName = GetDayName(row.DayOfWeek),
            StartTime = row.StartTime.ToString(@"hh\:mm"),
            EndTime = row.EndTime.ToString(@"hh\:mm"),
            IsAvailable = row.IsAvailable,
            SlotDurationMinutes = row.SlotDurationMinutes > 0 ? row.SlotDurationMinutes : 30,
            TimeSlots = GenerateTimeSlots(row.StartTime, row.EndTime, row.SlotDurationMinutes > 0 ? row.SlotDurationMinutes : 30)
        };
    }

    private static List<string> GenerateTimeSlots(TimeSpan start, TimeSpan end, int durationMinutes)
    {
        var slots = new List<string>();
        var current = start;

        // Ensure duration is valid to avoid infinite loop
        if (durationMinutes <= 0) durationMinutes = 30;

        while (current <= end)
        {
            slots.Add(current.ToString(@"hh\:mm"));
            current = current.Add(TimeSpan.FromMinutes(durationMinutes));
        }

        return slots;
    }

    private async Task<Dictionary<int, bool>> GetScheduleMapAsync(IEnumerable<int> doctorIds)
    {
        var ids = doctorIds.Distinct().ToList();
        if (ids.Count == 0)
        {
            return new Dictionary<int, bool>();
        }

        var availabilities = await _unitOfWork.Repository<DoctorAvailability>()
            .FindAsync(availability => ids.Contains(availability.DoctorId));

        return availabilities
            .GroupBy(availability => availability.DoctorId)
            .ToDictionary(group => group.Key, group => group.Any(row => row.IsAvailable));
    }

    private static bool IsProfileComplete(Doctor doctor)
    {
        return !string.IsNullOrWhiteSpace(doctor.Bio)
            && !string.IsNullOrWhiteSpace(doctor.ImageUrl)
            && !string.Equals(doctor.ImageUrl, "default-doctor.png", StringComparison.OrdinalIgnoreCase);
    }

    private static string GetDayName(byte dayOfWeek)
    {
        return CultureInfo.InvariantCulture.DateTimeFormat.DayNames[dayOfWeek];
    }

    private static TimeSpan ParseTime(string value, TimeSpan fallback)
    {
        return TimeSpan.TryParse(value, out var parsed) ? parsed : fallback;
    }

    private List<Doctor> FilterActiveDoctors(IEnumerable<Doctor> doctors)
    {
        return doctors.Where(IsDoctorUserActive).ToList();
    }

    private static bool IsDoctorUserActive(Doctor doctor)
    {
        if (!doctor.UserId.HasValue)
        {
            return false;
        }

        return doctor.User is { IsActive: true, IsDeleted: false };
    }

    private async Task<Doctor?> GetDoctorByUserIdAsync(int userId, bool includeUser = false, bool includeSpecialty = false)
    {
        if (userId <= 0)
        {
            return null;
        }

        var includes = new List<Expression<Func<Doctor, object>>>();
        if (includeUser) includes.Add(d => d.User!);
        if (includeSpecialty) includes.Add(d => d.Specialty);

        var doctors = includes.Count > 0
            ? await _unitOfWork.Repository<Doctor>().FindAsync(d => d.UserId == userId, includes.ToArray())
            : await _unitOfWork.Repository<Doctor>().FindAsync(d => d.UserId == userId);

        return doctors.FirstOrDefault();
    }

    private static DateTime? ParseAppointmentDateTime(string date, string time)
    {
        // Try various common formats
        string[] formats = { "yyyy-MM-dd", "dd MMM yyyy", "d MMM yyyy", "dd-MM-yyyy", "MM/dd/yyyy" };

        if (!DateTime.TryParse($"{date} {time}", out var combined))
        {
            if (!DateTime.TryParse(date, out combined))
            {
                // Last ditch effort: try cleaning the strings
                return null;
            }
        }

        return combined;
    }

    private static AppointmentDto MapAppointment(Appointment appointment)
    {
        var isFreeRebook = !string.IsNullOrWhiteSpace(appointment.Notes)
            && appointment.Notes.Contains("[FREE_REBOOK]", StringComparison.OrdinalIgnoreCase);

        return new AppointmentDto
        {
            Id = appointment.Id,
            PatientId = appointment.PatientId,
            DoctorId = appointment.DoctorId,
            PatientName = appointment.Patient?.FullName ?? string.Empty,
            DoctorName = appointment.Doctor?.Name ?? string.Empty,
            Specialty = appointment.Doctor?.Specialty?.Name ?? string.Empty,
            Date = appointment.Date,
            Time = appointment.Time,
            ScheduledAt = ParseAppointmentDateTime(appointment.Date, appointment.Time)?.ToString("O") ?? $"{appointment.Date} {appointment.Time}",
            PaymentMethod = appointment.PaymentMethod,
            Status = appointment.Status,
            Notes = appointment.Notes,
            IsFreeRebook = isFreeRebook,
            CanRebook = string.Equals(appointment.Status, "Missed", StringComparison.OrdinalIgnoreCase)
        };
    }
}
