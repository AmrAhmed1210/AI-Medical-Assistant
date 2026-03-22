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

        public AppointmentService(IUnitOfWork unitOfWork)
        {
            _unitOfWork = unitOfWork;
        }

        public async Task<AppointmentDto> CreateAppointmentAsync(CreateAppointmentDto dto)
        {
            var appointment = new Appointment
            {
                PatientId = dto.PatientId,
                DoctorId = dto.DoctorId,
                Date = dto.Date,
                Time = dto.Time,
                PaymentMethod = dto.PaymentMethod,
                Status = "confirmed",
                Notes = dto.Notes,
                CreatedAt = DateTime.UtcNow
            };

            await _unitOfWork.Appointments.AddAsync(appointment);
            await _unitOfWork.SaveChangesAsync();

            var saved = await _unitOfWork.Appointments.GetByIdWithDoctorAsync(appointment.Id);
            return MapToDto(saved ?? appointment);
        }

        public async Task<AppointmentDto?> GetAppointmentByIdAsync(int id)
        {
            var appointment = await _unitOfWork.Appointments.GetByIdWithDoctorAsync(id);
            return appointment == null ? null : MapToDto(appointment);
        }

        public async Task<IEnumerable<AppointmentDto>> GetAppointmentsByPatientIdAsync(int patientId)
        {
            var appointments = await _unitOfWork.Appointments.GetByPatientIdWithDoctorAsync(patientId);
            return appointments.Select(MapToDto);
        }

        public async Task<IEnumerable<AppointmentDto>> GetAppointmentsByDoctorIdAsync(int doctorId)
        {
            var appointments = await _unitOfWork.Appointments.GetByDoctorIdAsync(doctorId);
            return appointments.Select(MapToDto);
        }

        public async Task<AppointmentDto?> UpdateAppointmentAsync(UpdateAppointmentDto dto)
        {
            var appointment = await _unitOfWork.Appointments.GetByIdAsync(dto.Id);
            if (appointment == null) return null;

            appointment.Status = dto.Status;
            appointment.Notes = dto.Notes;

            _unitOfWork.Appointments.Update(appointment);
            await _unitOfWork.SaveChangesAsync();
            return MapToDto(appointment);
        }

        public async Task<bool> DeleteAppointmentAsync(int id)
        {
            var appointment = await _unitOfWork.Appointments.GetByIdAsync(id);
            if (appointment == null) return false;

            _unitOfWork.Appointments.Delete(appointment);
            await _unitOfWork.SaveChangesAsync();
            return true;
        }

        public async Task<PaginatedResultDto<AppointmentDto>> GetPaginatedAppointmentsAsync(int pageNumber, int pageSize)
        {
            var (items, totalCount) = await _unitOfWork.Appointments.GetPaginatedAsync(pageNumber, pageSize);
            return new PaginatedResultDto<AppointmentDto>
            {
                Items = items.Select(MapToDto),
                TotalCount = totalCount,
                PageNumber = pageNumber,
                PageSize = pageSize
            };
        }

        private static AppointmentDto MapToDto(Appointment a)
        {
            return new AppointmentDto
            {
                Id = a.Id,
                PatientId = a.PatientId,
                DoctorId = a.DoctorId,
                DoctorName = a.Doctor?.Name ?? string.Empty,
                Specialty = a.Doctor?.Specialty?.Name ?? string.Empty,
                Date = a.Date,
                Time = a.Time,
                PaymentMethod = a.PaymentMethod,
                Status = a.Status,
                Notes = a.Notes
            };
        }
    }
}