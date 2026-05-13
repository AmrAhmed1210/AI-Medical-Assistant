using MedicalAssistant.Domain.Entities.PatientModule;
using System.Collections.Generic;

namespace MedicalAssistant.Services_Abstraction.DTOs
{
    public class PatientCompleteHistoryDTO
    {
        public IEnumerable<ChronicDiseaseMonitor> ChronicDiseases { get; set; } = new List<ChronicDiseaseMonitor>();
        public IEnumerable<AllergyRecord> Allergies { get; set; } = new List<AllergyRecord>();
        public IEnumerable<MedicationTracker> Medications { get; set; } = new List<MedicationTracker>();
    }
}
