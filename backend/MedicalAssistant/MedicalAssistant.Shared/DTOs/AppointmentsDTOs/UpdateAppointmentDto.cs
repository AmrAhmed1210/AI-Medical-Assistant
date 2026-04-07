using System.ComponentModel.DataAnnotations;

namespace MedicalAssistant.Shared.DTOs.AppointmentsDTOs
{
    public class UpdateAppointmentDto
    {
        [Required]
        public int Id { get; set; }
        [Required]
        public int PatientId { get; set; }
        [Required]
        public int DoctorId { get; set; }
        public int? SessionId { get; set; }
        [Required]
        public DateTime ScheduledAt { get; set; }
        [Required]
        public string Status { get; set; } = "Pending";
        public string? Reason { get; set; }
        public string? Notes { get; set; }
        public bool IsDeleted { get; set; }
        public DateTime? UpdatedAt { get; set; }
    }
}
