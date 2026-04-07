namespace MedicalAssistant.Shared.DTOs.AppointmentsDTOs
{
    public class AppointmentDto
    {
        public int Id { get; set; }
        public int PatientId { get; set; }
        public int DoctorId { get; set; }
        public int? SessionId { get; set; }
        public DateTime ScheduledAt { get; set; }
        public string Status { get; set; } = "Pending";
        public string? Reason { get; set; }
        public string? Notes { get; set; }
        public bool IsDeleted { get; set; }
        public DateTime CreatedAt { get; set; }
        public DateTime? UpdatedAt { get; set; }
    }
}
