using AutoMapper;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.AppointmentsModule;
using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Domain.Entities.AnalysisModule;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.AppointmentsDTOs;
using MedicalAssistant.Shared.DTOs.DoctorDTOs;
using MedicalAssistant.Shared.DTOs.PatientDTOs;
using Microsoft.AspNetCore.Http;

namespace MedicalAssistant.Application.Services;

public class DoctorService : IDoctorService
{
    private readonly IUnitOfWork _unitOfWork;
    private readonly IMapper _mapper;

    public DoctorService(IUnitOfWork unitOfWork, IMapper mapper)
    {
        _unitOfWork = unitOfWork;
        _mapper = mapper;
    }

    public async Task<IReadOnlyList<DoctorDTO>> GetAllDoctorsAsync()
    {
        var doctors = await _unitOfWork.Repository<Doctor>().GetAllAsync();
        return _mapper.Map<IReadOnlyList<DoctorDTO>>(doctors);
    }

    public async Task<DoctorDetailsDTO?> GetDoctorByIdAsync(int id)
    {
        var doctor = await _unitOfWork.Repository<Doctor>().GetByIdAsync(id);
        return doctor is null ? null : _mapper.Map<DoctorDetailsDTO>(doctor);
    }

    public async Task<IReadOnlyList<DoctorDTO>> GetAvailableDoctorsAsync()
    {
        var doctors = await _unitOfWork.Doctors.GetAvailableDoctorsAsync();
        return _mapper.Map<IReadOnlyList<DoctorDTO>>(doctors);
    }

    public async Task<IReadOnlyList<DoctorDTO>> GetDoctorsBySpecialtyAsync(int specialtyId)
    {
        var doctors = await _unitOfWork.Doctors.GetBySpecialtyAsync(specialtyId);
        return _mapper.Map<IReadOnlyList<DoctorDTO>>(doctors);
    }

    public async Task<IReadOnlyList<DoctorDTO>> SearchDoctorsAsync(string name)
    {
        var doctors = await _unitOfWork.Doctors.SearchByNameAsync(name);
        return _mapper.Map<IReadOnlyList<DoctorDTO>>(doctors);
    }

    public async Task<IReadOnlyList<DoctorDTO>> GetTopRatedDoctorsAsync(int count)
    {
        var doctors = await _unitOfWork.Doctors.GetTopRatedDoctorsAsync(count);
        return _mapper.Map<IReadOnlyList<DoctorDTO>>(doctors);
    }

    public async Task<(IReadOnlyList<DoctorDTO> Items, int TotalCount)> GetPaginatedDoctorsAsync(int pageNumber, int pageSize)
    {
        var (items, totalCount) = await _unitOfWork.Doctors.GetPaginatedAsync(pageNumber, pageSize);
        return (_mapper.Map<IReadOnlyList<DoctorDTO>>(items), totalCount);
    }

    public async Task<DoctorDashboardDto> GetDoctorDashboardAsync(int doctorId)
    {
        var appointments = await _unitOfWork.Repository<Appointment>().GetAllAsync();
        var doctorAppts = appointments.Where(a => a.DoctorId == doctorId).ToList();

        return new DoctorDashboardDto
        {
            TodayAppointments = doctorAppts.Count(a => a.ScheduledAt.Date == DateTime.UtcNow.Date),
            PendingAppointments = doctorAppts.Count(a => a.Status == "Pending"),
            TotalPatients = doctorAppts.Select(a => a.PatientId).Distinct().Count(),
            UnreadReports = 0
        };
    }

    public async Task<IEnumerable<AvailabilityDto>> GetAvailabilityAsync(int doctorId)
    {
        var slots = await _unitOfWork.Doctors.GetAvailabilityAsync(doctorId);
        return _mapper.Map<IEnumerable<AvailabilityDto>>(slots);
    }

    public async Task UpdateAvailabilityAsync(int doctorId, IEnumerable<AvailabilityDto> slots)
    {
        var entities = _mapper.Map<IEnumerable<DoctorAvailability>>(slots);
        foreach (var entity in entities) entity.DoctorId = doctorId;
        await _unitOfWork.Doctors.UpdateAvailabilityAsync(doctorId, entities);
    }

    public async Task UpdateProfileAsync(int doctorId, DoctorUpdateDto dto)
    {
        var doctor = await _unitOfWork.Repository<Doctor>().GetByIdAsync(doctorId);
        if (doctor != null)
        {
            _mapper.Map(dto, doctor);
            await _unitOfWork.SaveChangesAsync();
        }
    }

    public async Task<string> UploadProfilePhotoAsync(int doctorId, IFormFile file)
    {
        var fileName = $"{doctorId}_{Guid.NewGuid()}{Path.GetExtension(file.FileName)}";
        var path = Path.Combine("wwwroot/uploads/profiles", fileName);

        if (!Directory.Exists("wwwroot/uploads/profiles")) Directory.CreateDirectory("wwwroot/uploads/profiles");

        using (var stream = new FileStream(path, FileMode.Create))
        {
            await file.CopyToAsync(stream);
        }

        var doctor = await _unitOfWork.Repository<Doctor>().GetByIdAsync(doctorId);
        if (doctor != null)
        {
            doctor.ImageUrl = $"/uploads/profiles/{fileName}";
            await _unitOfWork.SaveChangesAsync();
        }

        return doctor?.ImageUrl ?? string.Empty;
    }

    public async Task<IEnumerable<AppointmentDto>> GetAppointmentsByDoctorAsync(int doctorId, string? status)
    {
        var appointments = await _unitOfWork.Repository<Appointment>().GetAllAsync();
        var filtered = appointments.Where(a => a.DoctorId == doctorId && (status == null || a.Status == status));
        return _mapper.Map<IEnumerable<AppointmentDto>>(filtered);
    }

    public async Task<IEnumerable<PatientDto>> GetPatientsByDoctorAsync(int doctorId, string? search)
    {
        var appointments = await _unitOfWork.Repository<Appointment>().GetAllAsync();
        var patientIds = appointments.Where(a => a.DoctorId == doctorId).Select(a => a.PatientId).Distinct();

        var allPatients = await _unitOfWork.Repository<Patient>().GetAllAsync();
        var filtered = allPatients.Where(p => patientIds.Contains(p.Id));

        if (!string.IsNullOrEmpty(search))
        {
            filtered = filtered.Where(p => p.FullName.Contains(search, StringComparison.OrdinalIgnoreCase));
        }

        return _mapper.Map<IEnumerable<PatientDto>>(filtered);
    }

    public async Task<IEnumerable<AIReportDto>> GetAIReportsAsync(int doctorId, string? urgency, Guid? patientId)
    {
        var reports = await _unitOfWork.Repository<AnalysisResult>().GetAllAsync();
        var filtered = reports.Where(r => (urgency == null || r.UrgencyLevel == urgency));
        return _mapper.Map<IEnumerable<AIReportDto>>(filtered);
    }
}