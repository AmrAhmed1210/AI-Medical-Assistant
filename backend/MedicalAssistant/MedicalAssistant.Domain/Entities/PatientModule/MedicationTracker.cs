using MedicalAssistant.Domain.Entities.DoctorsModule;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MedicalAssistant.Domain.Entities.PatientModule
{
    public class MedicationTracker : BaseEntity
    {
        public int PatientId { get; set; }
        public Patient? Patient { get; set; }

        public int? PrescribedByDoctorId { get; set; }
        public Doctor? PrescribedByDoctor { get; set; }

        public int? ChronicDiseaseMonitorId { get; set; }
        public ChronicDiseaseMonitor? ChronicDiseaseMonitor { get; set; }

        public string MedicationName { get; set; } = string.Empty;
        public string? GenericName { get; set; }
        public string Dosage { get; set; } = string.Empty;
        public string Form { get; set; } = string.Empty;
        public string Frequency { get; set; } = string.Empty;
        public int TimesPerDay { get; set; }
        public string DoseTimes { get; set; } = string.Empty;
        public DateOnly StartDate { get; set; }
        public DateOnly? EndDate { get; set; }
        public string? Instructions { get; set; }
        public int? PillsRemaining { get; set; }
        public int RefillThreshold { get; set; } = 7;
        public bool IsChronic { get; set; }
        public bool IsActive { get; set; } = true;

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public ICollection<MedicationLog> Logs { get; set; } = new List<MedicationLog>();
    }
}
