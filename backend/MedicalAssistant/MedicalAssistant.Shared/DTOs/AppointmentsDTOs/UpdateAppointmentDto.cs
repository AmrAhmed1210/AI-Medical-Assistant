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
        [Required]
        public DateTime AppointmentDate { get; set; }
        [Required]
        public TimeSpan AppointmentTime { get; set; }
        [Required]
        public string Status { get; set; } = "Pending";
        public string? Notes { get; set; }
    }
}
