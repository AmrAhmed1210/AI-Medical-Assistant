using System.ComponentModel.DataAnnotations;

namespace MedicalAssistant.Shared.DTOs.AppointmentsDTOs
{
    public class CreateAppointmentDto
    {
        [Required]
        public int PatientId { get; set; }
        [Required]
        public int DoctorId { get; set; }
        public int? SessionId { get; set; }
        [Required]
        public DateTime ScheduledAt { get; set; }
        public string? Notes { get; set; }
    }
}
