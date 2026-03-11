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
                AppointmentDate = dto.AppointmentDate,
                AppointmentTime = dto.AppointmentTime,
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
            appointment.AppointmentDate = dto.AppointmentDate;
            appointment.AppointmentTime = dto.AppointmentTime;
            appointment.Status = dto.Status;
            appointment.Notes = dto.Notes;
            _appointmentRepo.Update(appointment);
            await _unitOfWork.SaveChangesAsync();
            return MapToDto(appointment);
        }

        public async Task<bool> DeleteAppointmentAsync(int id)
        {
            var appointment = await _appointmentRepo.GetByIdAsync(id);
            if (appointment == null) return false;
            _appointmentRepo.Delete(appointment);
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

        private static AppointmentDto MapToDto(Appointment appointment)
        {
            return new AppointmentDto
            {
                Id = appointment.Id,
                PatientId = appointment.PatientId,
                DoctorId = appointment.DoctorId,
                AppointmentDate = appointment.AppointmentDate,
                AppointmentTime = appointment.AppointmentTime,
                Status = appointment.Status,
                Notes = appointment.Notes,
                CreatedAt = appointment.CreatedAt
            };
        }
    }
}
