using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.ConsultationsModule;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.ConsultationDTOs;

namespace MedicalAssistant.Services.Services
{
    public class ConsultationService : IConsultationService
    {
        private readonly IUnitOfWork _unitOfWork;
        private readonly INotificationService _notificationService;

        public ConsultationService(IUnitOfWork unitOfWork, INotificationService notificationService)
        {
            _unitOfWork = unitOfWork;
            _notificationService = notificationService;
        }

        public async Task<ConsultationDto> CreateConsultationAsync(int doctorId, CreateConsultationDto dto)
        {
            var consultation = new Consultation
            {
                DoctorId = doctorId,
                PatientId = dto.PatientId,
                Title = dto.Title,
                Description = dto.Description,
                ScheduledAt = dto.ScheduledAt.ToUniversalTime(),
                Status = "Scheduled",
                CreatedAt = DateTime.UtcNow
            };

            await _unitOfWork.Repository<Consultation>().AddAsync(consultation);
            await _unitOfWork.SaveChangesAsync();

            var saved = await _unitOfWork.Repository<Consultation>()
                .FindAsync(c => c.Id == consultation.Id);

            var result = saved.FirstOrDefault();
            if (result == null)
                throw new InvalidOperationException("Failed to retrieve saved consultation");

            // Notify patient via SignalR
            await _notificationService.NotifyNewConsultation(
                result.Patient?.Email ?? string.Empty,
                result.Doctor?.Name ?? "Doctor",
                result.Title,
                result.ScheduledAt.ToString("O"),
                result.Id);

            return MapToDto(result);
        }

        public async Task<ConsultationDto?> GetConsultationByIdAsync(int id)
        {
            var consultation = await _unitOfWork.Repository<Consultation>()
                .FindAsync(c => c.Id == id);
            
            var result = consultation.FirstOrDefault();
            if (result == null) return null;

            // Load related entities
            var patients = await _unitOfWork.Patients.GetByIdAsync(result.PatientId);
            var doctors = await _unitOfWork.Doctors.GetByIdAsync(result.DoctorId);
            
            result.Patient = patients;
            result.Doctor = doctors;

            return MapToDto(result);
        }

        public async Task<IEnumerable<ConsultationDto>> GetConsultationsByPatientIdAsync(int patientId)
        {
            var consultations = await _unitOfWork.Repository<Consultation>()
                .FindAsync(c => c.PatientId == patientId);

            return consultations.Select(MapToDto);
        }

        public async Task<IEnumerable<ConsultationDto>> GetConsultationsByDoctorIdAsync(int doctorId)
        {
            var consultations = await _unitOfWork.Repository<Consultation>()
                .FindAsync(c => c.DoctorId == doctorId);

            return consultations.Select(MapToDto);
        }

        public async Task<ConsultationDto?> UpdateConsultationAsync(UpdateConsultationDto dto)
        {
            var consultation = await _unitOfWork.Repository<Consultation>()
                .FindAsync(c => c.Id == dto.Id);
            
            var result = consultation.FirstOrDefault();
            if (result == null) return null;

            if (dto.Title != null)
                result.Title = dto.Title;
            if (dto.Description != null)
                result.Description = dto.Description;
            if (dto.ScheduledAt.HasValue)
                result.ScheduledAt = dto.ScheduledAt.Value.ToUniversalTime();
            if (dto.Status != null)
                result.Status = dto.Status;

            result.UpdatedAt = DateTime.UtcNow;

            _unitOfWork.Repository<Consultation>().Update(result);
            await _unitOfWork.SaveChangesAsync();

            return MapToDto(result);
        }

        public async Task<bool> DeleteConsultationAsync(int id)
        {
            var consultation = await _unitOfWork.Repository<Consultation>()
                .FindAsync(c => c.Id == id);
            
            var result = consultation.FirstOrDefault();
            if (result == null) return false;

            _unitOfWork.Repository<Consultation>().Delete(result);
            await _unitOfWork.SaveChangesAsync();

            return true;
        }

        public async Task<bool> CompleteConsultationAsync(int id)
        {
            var consultation = await _unitOfWork.Repository<Consultation>()
                .FindAsync(c => c.Id == id);
            
            var result = consultation.FirstOrDefault();
            if (result == null) return false;

            result.Status = "Completed";
            result.UpdatedAt = DateTime.UtcNow;

            _unitOfWork.Repository<Consultation>().Update(result);
            await _unitOfWork.SaveChangesAsync();

            return true;
        }

        public async Task<bool> CancelConsultationAsync(int id)
        {
            var consultation = await _unitOfWork.Repository<Consultation>()
                .FindAsync(c => c.Id == id);
            
            var result = consultation.FirstOrDefault();
            if (result == null) return false;

            result.Status = "Cancelled";
            result.UpdatedAt = DateTime.UtcNow;

            _unitOfWork.Repository<Consultation>().Update(result);
            await _unitOfWork.SaveChangesAsync();

            return true;
        }

        private static ConsultationDto MapToDto(Consultation consultation)
        {
            return new ConsultationDto
            {
                Id = consultation.Id,
                DoctorId = consultation.DoctorId,
                DoctorName = consultation.Doctor?.Name ?? consultation.Doctor?.User?.FullName ?? "Doctor",
                PatientId = consultation.PatientId,
                PatientName = consultation.Patient?.FullName ?? "Patient",
                Title = consultation.Title,
                Description = consultation.Description,
                ScheduledAt = consultation.ScheduledAt.ToString("O"),
                Status = consultation.Status,
                CreatedAt = consultation.CreatedAt.ToString("O")
            };
        }
    }
}
