using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MedicalAssistant.Shared.DTOs.ReviewDTOs
{
    public class CreateReviewDTO
    {
        public int DoctorId { get; set; }
        public int Rating { get; set; }
        public string Comment { get; set; } = string.Empty;
        public string Author { get; set; } = string.Empty;
        public string? PatientName { get; set; }
        public int? PatientId { get; set; }
    }
}
