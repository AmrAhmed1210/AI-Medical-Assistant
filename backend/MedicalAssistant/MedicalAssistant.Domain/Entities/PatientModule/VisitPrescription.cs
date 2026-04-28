using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MedicalAssistant.Domain.Entities.PatientModule
{
    public class VisitPrescription : BaseEntity
    {
        public int PatientVisitId { get; set; }
        public PatientVisit PatientVisit { get; set; } = null!;

        public string MedicationName { get; set; } = string.Empty;
        public string? GenericName { get; set; }
        public string Dosage { get; set; } = string.Empty;
        public string Form { get; set; } = string.Empty;
        public string Frequency { get; set; } = string.Empty;
        public int TimesPerDay { get; set; }
        public string? SpecificTimes { get; set; }  // JSON: ["08:00","20:00"]
        public string? Duration { get; set; }        // "30 days"
        public int? Quantity { get; set; }
        public string? Instructions { get; set; }
        public bool IsChronic { get; set; }          // لو true → بيتعمل MedicationTracker أوتوماتيك
        public int Refills { get; set; } = 0;
        public bool IsDispensed { get; set; } = false;
        public string? Notes { get; set; }

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    }
}
