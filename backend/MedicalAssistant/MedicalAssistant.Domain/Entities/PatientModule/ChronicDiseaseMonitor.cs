using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MedicalAssistant.Domain.Entities.PatientModule
{
    public class ChronicDiseaseMonitor : BaseEntity
    {
        public int PatientId { get; set; }
        public Patient? Patient { get; set; }

        public string DiseaseName { get; set; } = string.Empty;
        public string DiseaseType { get; set; } = string.Empty;
        public DateOnly? DiagnosedDate { get; set; }
        public string Severity { get; set; } = string.Empty;
        public bool IsActive { get; set; } = true;
        public string? DoctorNotes { get; set; }
        public string? TargetValues { get; set; }
        public string MonitoringFrequency { get; set; } = string.Empty;
        public DateOnly? LastCheckDate { get; set; }

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
        public ICollection<VitalReading> VitalReadings { get; set; } = new List<VitalReading>();
        public ICollection<MedicationTracker> Medications { get; set; } = new List<MedicationTracker>();
    }
}
