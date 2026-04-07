using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.AppointmentsModule;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.AppointmentsDTOs;
using MedicalAssistant.Shared.DTOs.PatientDTOs;

namespace MedicalAssistant.Services.Services
{
    public class AppointmentService : IAppointmentService
    {
        private readonly IUnitOfWork _unitOfWork;
        private readonly IAppointmentRepository _appointmentRepo;

        public AppointmentService(IUnitOfWork unitOfWork)
        {
            _unitOfWork = unitOfWork;
            _appointmentRepo = (IAppointmentRepository)_unitOfWork.GetType().GetProperty("Appointments")?.GetValue(_unitOfWork);
        }

        public async Task<AppointmentDto> CreateAppointmentAsync(CreateAppointmentDto dto)
        {
            var appointment = new Appointment
            {
                PatientId = dto.PatientId,
                DoctorId = dto.DoctorId,
                SessionId = dto.SessionId,
                ScheduledAt = dto.ScheduledAt,
                Status = "Pending",
                Notes = dto.Notes,
                CreatedAt = DateTime.UtcNow
            };
            await _appointmentRepo.AddAsync(appointment);
            await _unitOfWork.SaveChangesAsync();
            return MapToDto(appointment);
        }

        public async Task<AppointmentDto?> GetAppointmentByIdAsync(int id)
        {
            var appointment = await _appointmentRepo.GetByIdAsync(id);
            return appointment == null ? null : MapToDto(appointment);
        }

        public async Task<IEnumerable<AppointmentDto>> GetAppointmentsByPatientIdAsync(int patientId)
        {
            var appointments = await _appointmentRepo.GetByPatientIdAsync(patientId);
            return appointments.Select(MapToDto);
        }

        public async Task<IEnumerable<AppointmentDto>> GetAppointmentsByDoctorIdAsync(int doctorId)
        {
            var appointments = await _appointmentRepo.GetByDoctorIdAsync(doctorId);
            return appointments.Select(MapToDto);
        }

        public async Task<AppointmentDto?> UpdateAppointmentAsync(UpdateAppointmentDto dto)
        {
            var appointment = await _appointmentRepo.GetByIdAsync(dto.Id);
            if (appointment == null) return null;
            appointment.PatientId = dto.PatientId;
            appointment.DoctorId = dto.DoctorId;
            appointment.SessionId = dto.SessionId;
            appointment.ScheduledAt = dto.ScheduledAt;
            appointment.Status = dto.Status;
            appointment.Reason = dto.Reason;
            appointment.Notes = dto.Notes;
            appointment.IsDeleted = dto.IsDeleted;
            appointment.UpdatedAt = dto.UpdatedAt ?? DateTime.UtcNow;
            _appointmentRepo.Update(appointment);
            await _unitOfWork.SaveChangesAsync();
            return MapToDto(appointment);
        }

        public async Task<bool> DeleteAppointmentAsync(int id)
        {
            var appointment = await _appointmentRepo.GetByIdAsync(id);
            if (appointment == null) return false;
            // soft delete
            appointment.IsDeleted = true;
            appointment.UpdatedAt = DateTime.UtcNow;
            _appointmentRepo.Update(appointment);
            await _unitOfWork.SaveChangesAsync();
            return true;
        }

        public async Task<PaginatedResultDto<AppointmentDto>> GetPaginatedAppointmentsAsync(int pageNumber, int pageSize)
        {
            var (items, totalCount) = await _appointmentRepo.GetPaginatedAsync(pageNumber, pageSize);
            return new PaginatedResultDto<AppointmentDto>
            {
                Items = items.Select(MapToDto),
                TotalCount = totalCount,
                PageNumber = pageNumber,
                PageSize = pageSize
            };
        }

        // New operations: confirm, cancel, complete
        public async Task<AppointmentDto?> ConfirmAppointmentAsync(int id)
        {
            var appt = await _appointmentRepo.GetByIdAsync(id);
            if (appt == null) return null;
            appt.Status = "Confirmed";
            appt.UpdatedAt = DateTime.UtcNow;
            _appointmentRepo.Update(appt);
            await _unitOfWork.SaveChangesAsync();
            return MapToDto(appt);
        }

        public async Task<AppointmentDto?> CancelAppointmentAsync(int id, string reason)
        {
            var appt = await _appointmentRepo.GetByIdAsync(id);
            if (appt == null) return null;
            appt.Status = "Cancelled";
            appt.Reason = reason;
            appt.UpdatedAt = DateTime.UtcNow;
            _appointmentRepo.Update(appt);
            await _unitOfWork.SaveChangesAsync();
            return MapToDto(appt);
        }

        public async Task<AppointmentDto?> CompleteAppointmentAsync(int id, string? notes)
        {
            var appt = await _appointmentRepo.GetByIdAsync(id);
            if (appt == null) return null;
            appt.Status = "Completed";
            appt.Notes = notes ?? appt.Notes;
            appt.UpdatedAt = DateTime.UtcNow;
            _appointmentRepo.Update(appt);
            await _unitOfWork.SaveChangesAsync();
            return MapToDto(appt);
        }

        private static AppointmentDto MapToDto(Appointment appointment)
        {
            return new AppointmentDto
            {
                Id = appointment.Id,
                PatientId = appointment.PatientId,
                DoctorId = appointment.DoctorId,
                SessionId = appointment.SessionId,
                ScheduledAt = appointment.ScheduledAt,
                Status = appointment.Status,
                Reason = appointment.Reason,
                Notes = appointment.Notes,
                IsDeleted = appointment.IsDeleted,
                CreatedAt = appointment.CreatedAt,
                UpdatedAt = appointment.UpdatedAt
            };
        }
    }
}
