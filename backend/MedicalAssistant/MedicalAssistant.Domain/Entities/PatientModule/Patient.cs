using MedicalAssistant.Domain.Entities.UserModule;

namespace MedicalAssistant.Domain.Entities.PatientModule
{
    public class Patient : BaseEntity
    {
        public string FullName { get; set; } = default!;
        public string Email { get; set; } = default!;
        public string PhoneNumber { get; set; } = default!;
        public string PasswordHash { get; set; } = string.Empty;
        public DateTime DateOfBirth { get; set; }
        public string Gender { get; set; } = default!;
        public string? Address { get; set; }
        public string? ImageUrl { get; set; }
        public string? BloodType { get; set; }
        public string? MedicalNotes { get; set; }
        public DateTime CreatedAt { get; set; }
        public bool IsActive { get; set; }
        public int? UserId { get; set; }
        public virtual User? User { get; set; }

        public virtual MedicalProfile? MedicalProfile { get; set; }
        public virtual ICollection<SurgeryHistory> SurgeryHistories { get; set; } = [];
        public virtual ICollection<AllergyRecord> AllergyRecords { get; set; } = [];
        public virtual ICollection<ChronicDiseaseMonitor> ChronicDiseaseMonitors { get; set; } = [];
        public virtual ICollection<VitalReading> VitalReadings { get; set; } = [];
        public virtual ICollection<MedicationTracker> MedicationTrackers { get; set; } = [];
        public virtual ICollection<MedicationLog> MedicationLogs { get; set; } = [];
        public virtual ICollection<PatientVisit> PatientVisits { get; set; } = [];
        public virtual ICollection<VisitVitalSign> VisitVitalSigns { get; set; } = [];
    }
}